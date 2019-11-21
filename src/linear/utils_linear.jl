function compute_observation(context::Array, theta::Array, sigma::Real = 1)
    obs = dot(context, theta) + rand(Normal(0, sigma^2))[1]
end


function update_design_inverse(matrix::Array, context::Array)
    matrix = matrix -
             matrix * context * transpose(context) * matrix /
             (1 + transpose(context) * matrix * context)
    return matrix
end


function update_square_root(matrix::Array, context::Array)
    norm = transpose(context) * matrix * matrix * context
    matrix = matrix -
             (1 - sqrt(1 - 4 * norm / (1 + norm))) / (2 * norm) * matrix * context *
             tranpose(context) * matrix
    return matrix
end
