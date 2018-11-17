using Base
using Statistics
using Flux

train_images = Flux.Data.MNIST.images(:train)
train_labels = Flux.Data.MNIST.labels(:train)

train_data = gpu.([
    (float(cat(train_images..., dims=4)), Flux.onehotbatch(train_labels, 0:9)) 
    for (train_images, train_labels) in zip(Base.Iterators.partition(train_images, 100), Base.Iterators.partition(train_labels, 100))
])

test_images = Flux.Data.MNIST.images(:test)
test_labels = Flux.Data.MNIST.labels(:test)

test_data = gpu.((float(cat(test_images..., dims=4)), Flux.onehotbatch(test_labels, 0:9)))

model = Chain(
    Conv((3, 3), 1=>32, relu, pad=(1, 1), stride=(1, 1)),
    MaxPool((2,2), pad=(0, 0), stride=(2, 2)),
    Conv((3, 3), 32=>64, relu, pad=(1, 1), stride=(1, 1)),
    MaxPool((2,2), pad=(0, 0), stride=(2, 2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(7 * 7 * 64, 1024, relu),
    Dense(1024, 10),
    softmax
) |> gpu

loss(x, y) = Flux.crossentropy(model(x), y)
accuracy(x, y) = mean(Flux.onecold(model(x)) .== Flux.onecold(y))

Flux.@epochs 10 Flux.train!(loss, train_data, ADAM(params(model)), cb=Flux.throttle(() -> @show(accuracy(test_data...)), 10))
