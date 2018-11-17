using Base
using Statistics
using Flux
using Images

train_images = Flux.Data.MNIST.images(:train)
train_data = gpu.([(float(hcat(vec.(train_images)...)),) for train_images in Base.Iterators.partition(train_images, 100)])

test_images = Flux.Data.MNIST.images(:test)
test_data = gpu.((float(hcat(vec.(Flux.Data.MNIST.images(:test))...)),))

model = Chain(
    Dense(28 ^ 2, 64, relu),
    Dense(64, 32, relu),
    Dense(32, 64, relu), 
    Dense(64, 28 ^ 2, relu), 
) |> gpu

loss(x) = Flux.mse(model(x), x)

Flux.@epochs 10 Flux.train!(loss, train_data, ADAM(params(model)), cb=Flux.throttle(() -> @show(loss(test_data...)), 10))

function generate()
    image(x::Vector) = Gray.(reshape(clamp.(x, 0, 1), 28, 28))
    inputs = rand(test_images, 10)
    outputs = image.(map(x -> cpu(model)(float(vec(x))).data, inputs))
    hcat(vcat.(inputs, outputs)...)
end

cd(@__DIR__)
save("generated.png", generate())
