using Flux
using Random
using Statistics

# Random seed for reproducibility
Random.seed!(1234)

# Hyperparameters
latent_dim = 16      # Noise vector dimension
label_dim = 2        # One-hot label dimension
feature_dim = 32     # Output feature vector dimension
batch_size = 16
epochs = 50

# Generator: learns to map (noise + label) -> fake feature vector
generator = Chain(
    Dense(latent_dim + label_dim, 64, relu),
    Dense(64, feature_dim)
)

# Discriminator: learns to classify (feature + label) as real/fake
discriminator = Chain(
    Dense(feature_dim + label_dim, 64, relu),
    Dense(64, 1),
    σ
)

# Optimizers
gen_opt = Flux.Adam()
disc_opt = Flux.Adam()

# --- Toy data: 100 samples ---
num_samples = 100

# Random real feature vectors (simulating preprocessed tweet embeddings)
real_features = randn(Float32, feature_dim, num_samples)

# Random binary labels (0 or 1), converted to one-hot
raw_labels = rand(0:1, num_samples)
labels = Flux.onehotbatch(raw_labels, 0:1)

# --- Training Loop ---
println("🚀 Starting CGAN training...")

for epoch in 1:epochs
    epoch_d_loss = Float32[]
    epoch_g_loss = Float32[]
    
    # Shuffle indices
    indices = shuffle(1:num_samples)
    
    for i in 1:batch_size:num_samples
        batch_idx = indices[i:min(i+batch_size-1, num_samples)]
        
        real_batch = real_features[:, batch_idx]
        label_batch = labels[:, batch_idx]
        b_size = size(real_batch, 2)
        
        # 1️⃣ Train Discriminator
        z = randn(Float32, latent_dim, b_size)
        fake_batch = generator(vcat(z, label_batch))
        
        pred_real = discriminator(vcat(real_batch, label_batch))
        pred_fake = discriminator(vcat(fake_batch, label_batch))
        
        d_loss = -mean(log.(pred_real) .+ log.(1 .- pred_fake))
        push!(epoch_d_loss, d_loss)
        
        d_grads = gradient(Flux.params(discriminator)) do
            pred_real = discriminator(vcat(real_batch, label_batch))
            pred_fake = discriminator(vcat(fake_batch, label_batch))
            -mean(log.(pred_real) .+ log.(1 .- pred_fake))
        end
        
        Flux.Optimise.update!(disc_opt, Flux.params(discriminator), d_grads)
        
        # 2️⃣ Train Generator
        z = randn(Float32, latent_dim, b_size)
        
        g_grads = gradient(Flux.params(generator)) do
            fake_batch = generator(vcat(z, label_batch))
            pred_fake = discriminator(vcat(fake_batch, label_batch))
            -mean(log.(pred_fake))
        end
        
        Flux.Optimise.update!(gen_opt, Flux.params(generator), g_grads)
        
        # Compute generator loss for reporting
        fake_batch = generator(vcat(z, label_batch))
        pred_fake = discriminator(vcat(fake_batch, label_batch))
        g_loss = -mean(log.(pred_fake))
        push!(epoch_g_loss, g_loss)
    end
    
    println("Epoch $epoch/$epochs | D_loss = $(mean(epoch_d_loss)) | G_loss = $(mean(epoch_g_loss))")
end

println("✅ Training complete.")

# --- Example: Generate synthetic features ---
z_new = randn(Float32, latent_dim, 1)
label_positive = Flux.onehotbatch([1], 0:1)
fake_feature = generator(vcat(z_new, label_positive))
println("Example generated feature (positive sentiment):")
println(fake_feature)
