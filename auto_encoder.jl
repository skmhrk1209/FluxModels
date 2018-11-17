using Base
using Statistics
using Flux
using Images

train_images = Flux.Data.MNIST.images(:train)
train_data = gpu.([(float(hcat(vec.(train_images)...)),) for train_images in Base.Iterators.partition(train_images, 100)])

test_images = Flux.Data.MNIST.images(:test)
test_data = gpu.((float(hcat(vec.(test_images)...)),))

model = Chain(
    Dense(28 ^ 2, 64, relu),
    Dense(64, 32, relu),
    Dense(32, 64, relu), 
    Dense(64, 28 ^ 2, relu), 
) |> gpu

loss(x) = Flux.mse(model(x), x)

Flux.@epochs 10 Flux.train!(loss, train_data, ADAM(params(model)), cb=Flux.throttle(() -> @show(loss(test_data...)), 10))

compose(f, fs...) = isempty(fs) ? (xs...) -> f(xs) : (xs...) -> compose(fs...)(f(xs))

picked = rand(test_images, 10)
generated = map(compose(
    image -> cpu(model)(float(vec(image))), 
    vec -> Gray.(reshape(clamp.(vec), 0, 1), 28, 28)
), picked)

save("image.png", hcat(vcat.(picked, generated)...))
