function compute_observation(context::Array, theta::Array, sigma::Real = 1)
    obs = dot(context, theta) + rand(Normal(0, sigma^2))[1]
end


function compute_transportation(context1::Array, context2::Array, mu::Array, cov::Matrix)
    norm = transpose(context1 - context2) * cov * (context1 - context2)
    if (dot(context2, mu) - dot(context1, mu) < 0)
        cost = (dot(context1, mu) - dot(context2, mu))^2 / (2 * norm)
    else
        cost = 0
    end
    return cost
end


function compute_confidence(context1::Array, context2::Array, cov::Matrix)
    confidence = sqrt(transpose(context1 - context2) * cov * (context1 - context2))
    return confidence
end


function update_design(matrix::Array, context::Array)
    matrix = matrix + context * transpose(context)
    return matrix
end


function update_design_inverse(matrix::Array, context::Array)
    matrix = matrix -
             matrix * context * transpose(context) * matrix /
             (1 + transpose(context) * matrix * context)
    return matrix
end


function update_square_root(matrix::Array, context::Array)
    norm = transpose(context) * matrix * matrix * context
    matrix = matrix - (1 - sqrt(1 - norm / (1 + norm))) / norm * matrix * context *
             transpose(context) * matrix
    return matrix
end
