# Compute simple regrets
function compute_regrets(mu::AbstractArray{<:Real}, recommendations::AbstractArray{<:Integer}, budget::Integer)
	maxmu = argmax(mu)[1]
	regrets = zeros(1, budget)
	for i in 1:budget
		regrets[i] = maxmu - mu[recommendations[i]]
	end

	return regrets
end


# Compute simple regrets for a normalized reservoir
function compute_regrets_reservoir(mu::AbstractArray{<:Real}, recommendations::AbstractArray{<:Integer}, budget::Integer, maxmu::Float64 = 0.25)
	regrets = zeros(1, budget)
	for i in 1:budget
		regrets[i] = maxmu - mu[recommendations[i]]
	end

	return regrets
end


#############################
# Recommendation strategies #
#############################

# EDP: Empirical distribution of plays
function edp(N, S)
	total = sum(N)
	K = length(N)
	p = [N[i]/total for i in 1:K]
	d = Categorical(p)
	return rand(d)
end


# EBA: Empirical best arm
function eba(N, S)
	K = length(N)
	means = S ./ N
	idx = (LinearIndices(means .== maximum(means)))[findall(means .== maximum(means))]
	best = idx[floor(Int, length(idx) * rand()) + 1]
	return best
end


# MPA: Most played arm
function mpa(N, S)
	idx = (LinearIndices(N .== maximum(N)))[findall(N .== maximum(N))]
	best = idx[floor(Int, length(idx) * rand()) + 1]
	return best
end


function randmax(vector, rank=1)
   # returns an integer, not a CartesianIndex
   vector = vec(vector)
   Sorted = sort(vector,rev=true)
   m = Sorted[rank]
   Ind = findall(x->x==m,vector)
   index = Ind[floor(Int,length(Ind)*rand())+1]
   return index
end


# Memory usage check
# function memuse()
#   return string(round(Int, parse(Int, readstring(`ps -p 29563 -o rss=`))/1024), "M")
# end
