using Base
using Flux
using CuArrays

train_images = float.(Flux.Data.MNIST.images(:train))
train_labels = Flux.onehotbatch(Flux.Data.MNIST.labels(:train), 0:9)

test_images = float.(Flux.Data.MNIST.images(:test))
test_labels = Flux.onehotbatch(Flux.Data.MNIST.labels(:test), 0:9)

train_data = gpu.([
    (cat(train_images[indices]..., dims=4), train_labels[:,indices]) 
    for indices in Base.Iterators.partition(1:length(train_images), 100)
])

test_data = (cat(test_images..., dims=4), test_labels)

model = gpu(Chain(
   Conv((3, 3), 1=>32, relu, pad=(1, 1), stride=(1, 1)),
   MaxPool((2,2), pad=(0, 0), stride=(2, 2)),
   Conv((3, 3), 32=>64, relu, pad=(1, 1), stride=(1, 1)),
   MaxPool((2,2), pad=(0, 0), stride=(2, 2)),
   x -> reshape(x, :, size(x, 4)),
   Dense(7 * 7 * 64, 1024),
   Dense(1024, 10), 
   softmax
))

loss(x, y) = Flux.crossentropy(model(x), y)
accuracy(x, y) = mean(Flux.onecold(model(x)) .== Flux.onecold(y))

Flux.train!(loss, train, ADAM(params(model)), cb=Flux.throttle(() -> @show(accuracy(test_data...)), 10))
