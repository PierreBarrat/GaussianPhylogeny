include("MessagePassing.jl")
include("OT-Tree.jl")

type OTnode
	par::Int64
	children::Array{Int64,1}
	dt::Float64
	conf::Array{Float64,1}
end


"""
	function ComputeGaussPw(Aii::Array{Float64,2},Ajj::Array{Float64,2},Aij::Array{Float64,2},Bi::Array{Float64,1},Bj::Array{Float64,1})

Performs the following gaussian integration 
Z_{12} =\int\ddroit X_1 \ddroit X_2 \exp\left\{ X_1A_{11}X_1 + X_2A_{22}X_2 + X_1A_{12}X_2 + B_1X_1 + B_2X_2 \right\}
and returns the log of the scalar result.
"""
function ComputeGaussPw(Aii::Array{Float64,2},Ajj::Array{Float64,2},Aij::Array{Float64,2},Bi::Array{Float64,1},Bj::Array{Float64,1})

	K = -[2*Aii Aij ; Aij 2*Ajj]
	m = [Aij*Bj - 2*Ajj*Bi ; Aij*Bi - 2*Aii*Bj]
	# L = size(K)[1]
	# return sqrt((2*pi)^L/det(K))*exp(m'*K*m)
	return (-1/2*log(det(K)) + m'*inv(K)*m)
end

"""
"""
function ComputeGauss(A::Array{Float64,2},B::Array{Float64,1})
	return (-1/2*log(det(-A)) + B'*inv(-A)*B)
end

"""
"""
function MPComputeZij(MPgraph::Array{MPnode,1}, i::Int64, j::Int64, J::Array{Float64,2})

	mtoi = SumMessages(MPgraph[i],j)
	mtoj = SumMessages(MPgraph[j],i)

	return ComputeGaussPw(MPgraph[i].field + mtoi.Phi, MPgraph[j].field + mtoj.Phi, J, MPgraph[i].mu + mtoi.Psi, MPgraph[j].mu + mtoj.Psi)

end

"""
"""
function MPComputeZi(MPgraph::Array{MPnode,1}, i::Int64)
	mtoi = SumMessages(MPgraph[i],i)
	if i==1
		# println("MP --> Z1 = ",ComputeGauss(MPgraph[i].field + mtoi.Phi, MPgraph[i].mu + mtoi.Psi))
		# println("ComputeGauss($(MPgraph[i].field + mtoi.Phi),$(MPgraph[i].mu + mtoi.Psi))")
	end
	return ComputeGauss(MPgraph[i].field + mtoi.Phi, MPgraph[i].mu + mtoi.Psi)
end

"""
	function MPComputeF(MPgraph::Array{MPnode,1},J::Array{Float64,2})

Summing local partition functions of nodes, and pairwise partition functions for nodes and their parent, counting every pair once. 
"""
function MPComputeF(MPgraph::Array{MPnode,1},J::Array{Float64,2})
	N = size(MPgraph)[1]
	F = 0.
	for (i,node) in enumerate(MPgraph)
		r = size(node.children)[1] + Int64(node.par>0)
		F += (r-1) * MPComputeZi(MPgraph,i)
		if node.par > 0
			F -= MPComputeZij(MPgraph, i, MPgraph[i].par, J)
		end
		# println(F)
	end
	return F
end

"""
	function ComputeF(tree::Array{OTnode,1}, C::Array{Float64,2}, dt::Float64)

Computes free energy of the Ornstein-Uhlenbeck process on a tree `tree`. Potential is `invC`. No message passing is needed here.
"""
function ComputeF(tree::Array{OTnode,1}, invC::Array{Float64,2}, dt::Float64)
	N = size(tree)[1]
	L = size(C)[1]
	F = 0.
	Lam = expm(-dt*invC)
	iSig = inv(eye(L)-Lam^2)*invC
	fij = ComputeGaussPw(-1/2*iSig, -1/2*iSig, Lam*iSig, zeros(Float64, L), zeros(Float64, L))
	fi = -1/2*log(det(invC/2))
	# println("Ana --> fi = $fi")
	for node in tree
		r = size(node.children)[1] + Int64(node.par>0)
		F += (r-1)*fi
		if node.par >0
			F -= fij
		end
	end
	# return (F,iSig, Lam)
	return F
end


"""
	function InitMPgraph(tree::Array{OTnode,1}, invC::Array{Float64,2}, dt::Float64)

Create an `MPGraph` object, starting from the `OTnode` array `tree`, containing topology and configurations of (leaf) nodes, and from a correlation matrix `C`. Leaves are removed from `tree`, and fields `\mu` are computed from leaves configuration. Also returns the value of the dynamical coupling matrix `J`.
"""
function InitMPgraph(tree::Array{OTnode,1}, invC::Array{Float64,2}, dt::Float64)

	L = size(invC)[1]
	# Dynamical parameters
	Lam = expm(-dt*invC)
	iSig = inv(eye(L)-Lam^2)*invC
	J = Lam*iSig

	# Initializing MP tree
	N = size(tree)[1]
	Nmp = N - sum(map(x->Int64(isempty(x.children)),tree))
	MPgraph = Array{MPnode,1}(Nmp)
	maptree = zeros(Int64, Nmp)
	imaptree = zeros(Int64,N)
	imp=1
	for i in 1:N
		imaptree[i] = 0
		if !isempty(tree[i].children)
			MPgraph[imp] = MkMPnode()
			maptree[imp] = i
			imaptree[i] = imp
			imp+=1
		end
	end

	# Filling parents and children
	for i in 1:Nmp
		it = maptree[i]
		if tree[it].par > 0
			imaptree[tree[it].par]==0?error("GaussianPhylogeny.jl - InitMPgraph"):MPgraph[i].par = imaptree[tree[it].par]
		push!(MPgraph[MPgraph[i].par].children, i)
		else
			MPgraph[i].par = -1
		end
	end

	# Setting zero messages to children
	for i in 1:Nmp
		Nc = size(MPgraph[i].children,1)
		MPgraph[i].messfromchild = Array{message,1}(Nc)
		for k in 1:Nc
			MPgraph[i].messfromchild[k] = message(Array{Float64,2}(0,0), Array{Float64,1}(0), false)
		end		
	end

	# Setting mu's and fields
	for i in 1:Nmp
		if MPgraph[i].par == -1 # root
			MPgraph[i].field = -1/2 * (eye(L)+Lam^2)*iSig
		else # internal node -- leaves nodes have been removed in this case
			MPgraph[i].field = -1/2 * (eye(L)+2*Lam^2)*iSig
		end
		MPgraph[i].mu = zeros(Float64,L)
		it = maptree[i] # Are the children of the original tree node leaves ?
		for c in tree[it].children
			if isempty(tree[c].children) # This one is empty --> contributes to mu
				MPgraph[i].mu += (tree[c].conf' * Lam * iSig)'
			end
		end
	end

	return (MPgraph, J)
end

"""
"""
function ComputeLikelihoodLeaves(tree::Array{OTnode,1}, invC::Array{Float64,2}, dt::Float64)
	Lam = expm(-dt*invC)
	iSig = inv(eye(size(invC)[1])-Lam^2)*invC
	L = 0.
	for node in tree
		if isempty(node.children) 
			L += -1/2*node.conf' * iSig * node.conf
		end
	end
	return L
end


"""
	function ComputeLikelihood(tree::Array{OTnode,1}, invC::Array{Float64,2}, dt::Float64)

Given topology and samples contained in `tree`, and parameters `C`, computes value of the log-likelihood. Two different procedures for computing free energies are called: one using message passing, and the other analytical formula. 
"""
function ComputeLikelihood(tree::Array{OTnode,1}, invC::Array{Float64,2}, dt::Float64)
	F_all = ComputeF(tree, invC, dt)
	(MPgraph, J) = InitMPgraph(tree, invC, dt)
	ComputeMessages!(MPgraph,J)
	F_MP = MPComputeF(MPgraph, J)
	F_leaves = ComputeLikelihoodLeaves(tree, invC, dt)
	return (F_leaves + F_MP - F_all, F_all, F_MP, F_leaves, MPgraph, J)

end


# """
# 	function ReadTree(treetopo::String, msa::Array{Float64,2})
# """
# function ReadTree(treetopo::String, msa::Array{Float64,2})
# 	trin = readdlm(treetopo)
# 	N = size(trin)[1]
# 	tree = Array{OTnode,1}(N)
# 	for (i,n) in enumerate(trin[:,1])
# 		n = Int64(n)
# 		tree[n] = OTnode(-1, Array{Int64,1}(0), -1., Array{Float64,1}(0))
# 		tree[n].par = Int64(trin[i,2])
# 		if tree[n].par > 0
# 			push!(tree[tree[n].par].children,n)
# 		end
# 		tree[n].dt = trin[i,3]
# 	end
# 	return tree
# end