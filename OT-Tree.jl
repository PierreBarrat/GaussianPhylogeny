type OTnode
	par::Int64
	children::Array{Int64,1}
	dt::Float64
	conf::Array{Float64,1}
end

"""
	function GaussianVector(C::Array{Float64,2}, M::Int64)

Return `M` samples of gaussian vector of correlation matrix `C` and mean `mu`.
"""
function GaussianVector(C::Array{Float64,2}, mu::Array{Float64,1}, M::Int64)
	L = size(C)[1]
	Y = randn(Float64, M, L)
	X = Y*sqrtm(C) + repmat(mu',M,1)
	return X
end

"""
	function GaussianVector(C::Array{Float64,2}, M::Int64)

Return a realization of gaussian vector of correlation matrix `C` and mean `mu`.
"""
function GaussianVector(C::Array{Float64,2}, mu::Array{Float64,1})
	L = size(C)[1]
	Y = randn(Float64, L)
	X = sqrtm(C)*Y + mu
	return X
end

"""
	function ReadTree(treefile::String)

Read tree contained in `treefile` and return an array of `OTnode` type.
"""
function ReadTree(treefile::String)
	trin = readdlm(treefile)
	N = size(trin)[1]
	tree = Array{OTnode,1}(N)
	for (i,n) in enumerate(trin[:,1])
		n = Int64(n)
		tree[n] = OTnode(-1, Array{Int64,1}(0), -1., Array{Float64,1}(0))
		tree[n].par = Int64(trin[i,2])
		if tree[n].par > 0
			push!(tree[tree[n].par].children,n)
		end
		tree[n].dt = trin[i,3]
	end
	return tree
end

"""
"""
function SampleTree!(tree::Array{OTnode,1}, C::Array{Float64,2})
	L = size(C)[1]
	invC = inv(C)
	root = findin(map(x->x.par, tree),-1)[1]
	tree[root].conf = GaussianVector(C, zeros(Float64,L))
	SampleChildren!(tree,root,C,invC)
	return tree
end

"""
"""
function SampleChildren!(tree::Array{OTnode,1}, i::Int64,C::Array{Float64,2},invC::Array{Float64,2})
	# println("---- Sampling children of $i:")
	for c in tree[i].children
		SampleChild!(tree[c],tree[i],C,invC)
		SampleChildren!(tree,c,C,invC)
	end
end


"""
	function SampleChild(X1::OTnode, iSig::Array{Float64,2}, Lam::Array{Float64,2}, dt::Float64)

Sample a configuration initiating at `X1` and after time `dt`, with correlation matrix `C`. Ornstein Uhlenbeck process is used. 
"""
function SampleChild!(X2::OTnode, X1::OTnode, C::Array{Float64,2}, invC::Array{Float64,2})
	Lam = expm(-X2.dt*invC)
	mu = Lam*X1.conf
	X2.conf = GaussianVector((eye(Lam)-Lam^2)*C, mu)

end

"""
"""
function SampleTree(treefile::String, Cfile::String)
	C = readdlm(Cfile,Float64)
	tree = ReadTree(treefile)
	SampleTree!(tree, C)
	return tree
end


