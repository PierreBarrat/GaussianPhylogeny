type MPnode
	children::Array{Int64,1}
	par::Int64
	messfromchild::Array{message,1}
	messfrompar::message
	mu::Array{Float64,1}
	field::Array{Float64,2}
end

type message
	Phi::Array{Float64,2}
	Psi::Array{Float64,1}
	initialized::Bool
end



"""
	function AskMessage(MPgraph::Array{MPnode,1}, i::Int64, j::Int64, J::Array{Float64,2})

`i` asks for a message to `j`. The output should be the message from `j` to `i`. Function operates recursively as follows
1. `j` requires messages from all its children `k`, calling `AskMessage(MPgraph,j,k)`. If `i` is also a children of `k`, error is raised. If `j` has no children, this part has no effect.
2. Message from `j` to `i` is computed using the local fields at `j`, the coupling between `i` and `j`, and incoming messages from children to `j`. 
"""
function AskMessage(MPgraph::Array{MPnode,1}, i::Int64, j::Int64, J::Array{Float64,2})

	# Phi = zeros(Float64,q,q)
	# Psi = zeros(Float64,q)

	println("Asking message: $i ---> $j")
	# Safety checks
	if !isempty(findin(MPgraph[j].children, i))
		error("MessagePassing.jl - AskMessage: Asking message from a parent is not allowed.\n")
	end

	## !!!!  Recursive part !!!!  
	# If the graph topology is not a tree, this WILL NOT END 
	# Gathering messages from all children of `j`
	for (id,k) in enumerate(MPgraph[j].children)
		if !MPgraph[j].messfromchild[id].initialized
			MPgraph[j].messfromchild[id] = AskMessage(MPgraph, j, k, J)
		else
			error("MessagePassing.jl - AskMessage: Asking an already initialized message.\n")
		end
	end
	## !!!!  Recursive part !!!!  

	# Computing Phi and Psi - quadratic and linear parts of the message
	Phi = MPgraph[j].field
	Psi = MPgraph[j].mu
	for mkj in MPgraph[j].messfromchild
		if mkj.initialized
			Phi += mkj.Phi
			Psi += mkj.Psi
		else
			error("MessagePassing.jl - AskMessage: Operating on non initialized message.\n")
		end
	end
	Phi = inv(Phi)
	Psi = -1/2 * J * Phi * Psi
	Phi = -1/4 * J * Phi * J

	# Creating message and end
	return message(Phi, Psi, true)

end

""" 
	function SendMessage(MPgraph::Array{MPnode,1}, i::Int64, j::Int64)

Sends message from `i` to `j`. `j` should be one of the children of `i`. If this is not the case, the message should have previously been computed in the `AskMessage` step. 
1. Messages of all chidren of `i` except `j` and of the parent of `i` are grouped. They should all have initialized value of `true`.
2. Message from `i` to `j` is computed.  
"""
function SendMessage(MPgraph::Array{MPnode,1}, i::Int64, j::Int64, J::Array{Float64,2})

	# Phi = zeros(Float64,q,q)
	# Psi = zeros(Float64,q)

	println("Sending message: $i ---> $j")

	## Safety checks
	if isempty(findin(MPgraph[i].children,j))
		error("MessagePassing.jl - SendMessage: Can only send message to children.\n")
	end

	## Initializing Phi and Psi
	Phi = MPgraph[i].field
	Psi = MPgraph[i].mu
	# From children different of j
	for (k,mki) in enumerate(MPgraph[i].messfromchild)
		if MPgraph[i].children[k]!=j
			# println("---- msg $(MPgraph[i].children[k])-->$i")
			if mki.initialized
				Phi += mki.Phi
				Psi += mki.Psi
			else
				error("MessagePassing.jl - SendMessage: Operating on non initialized message: .\n")
			end
		end
	end
	# From par, if i is not the root
	if MPgraph[i].par>0
		if !MPgraph[i].messfrompar.initialized
			error("MessagePassing.jl - SendMessage: Operating on non initialized message.\n")
		else
			Phi += MPgraph[i].messfrompar.Phi
			Psi += MPgraph[i].messfrompar.Psi
		end
	end
	Phi = inv(Phi)
	Psi = -1/2 * J * Phi * Psi
	Phi = -1/4 * J * Phi * J

	return message(Phi,Psi,true)
end

"""
	function PushMessage(MPgraph::Array{MPnode,1}, i::Int64)

Computes and sends message from `i` to all of its children. Calls itself recursively on all children of `i`. 
"""
function PushMessage(MPgraph::Array{MPnode,1}, i::Int64, J::Array{Float64,2})
	# println("Pushing messages from $i.")
	for k in MPgraph[i].children
		MPgraph[k].messfrompar = SendMessage(MPgraph, i, k, J::Array{Float64,2})
		PushMessage(MPgraph, k, J)
	end
end

"""
	function ComputeMessages(MPgraph::Array{MPnode,1}, J::Array{Float64,2})

Compute all messages for the given tree `MPgraph`. Steps of the computation are the following:
1. Find the root of the tree, only node that does not have a parent.
2. Call `AskMessage(MPgraph, root, c, J)` for all `c` children of `root`. The `AskMessage` process is then called recursively on all the tree.
3. Call `PushMessage(MPgraph, root, J)`, pushing message from the root to all of its children, and calling `PushMessage` recursively.
"""
function ComputeMessages(MPgraph::Array{MPnode,1}, J::Array{Float64,2})

	## Finding root
	flag = 0
	root = 0 
	for (i,node) in enumerate(MPgraph)
		if node.par < 0
			root = i
			flag +=1
		end
	end
	if flag != 1
		error("MessagePassing.jl - ComputeMessages: Found $flag roots in the tree.\n")
	end

	## Starting the AskMessage procedure from roots
	for (id,c) in enumerate(MPgraph[root].children)
		MPgraph[root].messfromchild[id] = AskMessage(MPgraph, root, c, J)
	end

	## Sending messages from root 
	PushMessage(MPgraph, root, J)

	## Now MPgraph contains all messsages
	return MPgraph
end

"""
	function InitializeGraph(treefile::String, C::{Float64,2})

Initialize and return the message passing tree `MPgraph`. Topology is read in file `treefile`. Correlation matrix and \mu field parameters are inputs.
_*TO BE MODIFIED*_: does not allow different times for different branches. 
"""
function InitializeGraph(treefile::String, C::Array{Float64,2})

	# Reading tree file
	td = readdlm(treefile, Float64)
	dt = td[1,3]
	N = size(td,1)
	L = size(C,1)

	# Parameters of the dynamical 
	Lam = expm(-dt*inv(C))
	iSig = inv((eye(L)-Lam^2)*C)
	J = Lam*iSig

	# Initializing parents and children
	MPgraph = Array{MPnode,1}(N)
	for i in 1:N
		MPgraph[i] = MkMPnode()
		MPgraph[i].par = Int64(td[i,2])
		if td[i,2]>0
			push!(MPgraph[Int64(td[i,2])].children, i)
		end
	end

	# Initializing messages and parameters
	for i in 1:N
		
		# Messages
		Nc = size(MPgraph[i].children,1)
		MPgraph[i].messfromchild = Array{message,1}(Nc)
		for k in 1:Nc
			MPgraph[i].messfromchild[k] = message(Array{Float64,2}(0,0), Array{Float64,1}(0), false)
		end

		# Fields
		if MPgraph[i].par == -1 # root node
			MPgraph[i].field = -1/2 * (eye(L)+Lam^2)*iSig
		elseif isempty(MPgraph[i].children) # leaf
			MPgraph[i].field = -1/2 * iSig
		else # internal node
			MPgraph[i].field = -1/2 * (eye(L)+2*Lam^2)*iSig
		end

		# mus
		MPgraph[i].mu = zeros(Float64,L)

	end
	return (MPgraph, J)
end

"""
	function MkMPnode()

Return empty `MPnode` type variable. 
"""
function MkMPnode()
	return MPnode( Array{Int64,1}(0), -1,  Array{message,1}(0), message(Array{Float64,2}(0,0), Array{Float64,1}(0), false), Array{Float64,1}(0), Array{Float64,2}(0,0))
end

"""
"""
function RunMP(treefile::String, Cfile::String)

	C = readdlm(Cfile, Float64)

	(MPgraph, J) = InitializeGraph(treefile, C)
	typeof(MPgraph)
	MPgraph = ComputeMessages(MPgraph, J)
	return MPgraph
end

"""
"""
function ComputeSingleBelief(X::Array{Float64,1}, MPgraph::Array{MPnode,1}, i::Int64)
	bi = exp(X' * MPgraph[i].field * X + MPgraph[i].mu'*X)
	for mki in MPgraph[i].messfromchild
		bi *=  exp(X'*mki.Phi*X + mki.Psi'*X)
	end
	if MPgraph[i].par >0
		bi *= exp(X'*MPgraph[i].messfrompar.Phi*X + MPgraph[i].messfrompar.Psi'*X)
	end
	return bi
end

"""
	function ComputePairBelief(X::Array{Float64,1}, Y::Array{Float64,1}, MPgraph::Array{MPnode,1}, i::Int64, j::Int64, J)

`i` and `j` should be neighbours in the graph. 
"""
function ComputePairBelief(X::Array{Float64,1}, Y::Array{Float64,1}, MPgraph::Array{MPnode,1}, i::Int64, j::Int64, J)

	bij = exp(X' * J * Y)
	bij *= exp(X' * MPgraph[i].field * X + MPgraph[i].mu'*X + Y' * MPgraph[j].field * Y + MPgraph[j].mu'*Y)

	for (k,mki) in enumerate(MPgraph[i].messfromchild)
		if MPgraph[i].children[k]!=j
			bij *= exp(X'*mki.Phi*X + mki.Psi'*X)
		end
	end
	if MPgraph[i].par > 0 && MPgraph[i].par!=j
		bij *= exp(X'*MPgraph[i].messfrompar.Phi*X + MPgraph[i].messfrompar.Psi'*X)
	end

	for (k,mkj) in enumerate(MPgraph[j].messfromchild)
		if MPgraph[j].children[k]!=i
			bij *= exp(Y'*mkj.Phi*Y + mkj.Psi'*Y)
		end
	end
	if MPgraph[j].par > 0 && MPgraph[j].par!=i
		bij *= exp(Y'*MPgraph[j].messfrompar.Phi*Y + MPgraph[j].messfrompar.Psi'*Y)
	end

	return bij

end

"""
"""
function ComputeSingleFreq(X::Array{Float64,1}, C)
	return exp(-1/2 * X' * inv(C) * X)
end


"""
"""
function ComputePairFreq(X::Array{Float64,1}, Y::Array{Float64,1}, C, dt::Float64)
	Lam = expm(-dt*inv(C))
	iSig = inv((eye(size(C)[1])-Lam^2)*C)
	return exp(-1/2 * (X'*iSig*X + Y'*iSig*Y - 2*X'*Lam*iSig*Y))
end


