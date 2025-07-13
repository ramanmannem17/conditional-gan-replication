using Flux
using Statistics

# Simulate fake "real data" (like tweets embedded as vectors)
X_real = rand(Float32, 10, 16)  # 10 features, 16 samples
y_real = ones(Float32, 1, 16)   # all ones (real labels)

latent_dim = 5  # size of random noise vector

# Generator: latent_dim + label => data
generator = Chain(
    Dense(latent_dim + 1, 10, relu)
)

# Discriminator: data + label => real/fake
discriminator = Chain(
    Dense(11, 32, relu),
    Dense(32, 1),
    sigmoid
)

# Loss function
bce_loss(pred, target) = Flux.Losses.binarycrossentropy(pred, target)

# Setup optimizers
opt_g = Flux.setup(Adam(0.001), generator)
opt_d = Flux.setup(Adam(0.001), discriminator)

# Training loop
for epoch in 1:5
    # === Discriminator step ===
    z = randn(Float32, latent_dim, 16)
    labels = rand(Float32, 1, 16)
    fake_data = generator(vcat(z, labels))

    real_input = vcat(X_real, labels)
    fake_input = vcat(fake_data, labels)

    function d_loss()
        pred_real = discriminator(real_input)
        pred_fake = discriminator(fake_input)
        bce_loss(pred_real, ones(Float32, 1, 16)) +
        bce_loss(pred_fake, zeros(Float32, 1, 16))
    end

    # Compute gradients and update discriminator
    grads_d = gradient(Flux.trainable(discriminator)) do
        d_loss()
    end
    Flux.Optimise.update!(opt_d, Flux.trainable(discriminator), grads_d)

    # === Generator step ===
    z = randn(Float32, latent_dim, 16)
    labels = rand(Float32, 1, 16)

    function g_loss()
        preds = discriminator(vcat(generator(vcat(z, labels)), labels))
        bce_loss(preds, ones(Float32, 1, 16))
    end

    # Compute gradients and update generator
    grads_g = gradient(Flux.trainable(generator)) do
        g_loss()
    end
    Flux.Optimise.update!(opt_g, Flux.trainable(generator), grads_g)

    println("Epoch $epoch done. D loss: $(d_loss()) G loss: $(g_loss())")
end

println("Training complete.")
