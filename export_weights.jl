# ==========================================
# Export weights to JSON format for JavaScript
# ==========================================
println("Begin file...")
using JLD2, JSON3

println("Loading weights...")
@load joinpath(@__DIR__, "Harry_Potter_weights_training.jld2") params stoi itos merges vocab

W_embed, W_Q, W_K, W_V, W_O, W_ff1, b_ff1, W_ff2, b_ff2, W_out, b_out = params.components
print("✓ Weights loaded.\n")
print("W_embed size: ", size(W_embed), "\n")  # Print first 5 embeddings as a sample

matrix_to_rows(M) = [collect(M[i, :]) for i in 1:size(M, 1)]

weights_export = Dict(
    "params" => Dict(
        "W_embed" => matrix_to_rows(W_embed),
        "W_Q"     => matrix_to_rows(W_Q),
        "W_K"     => matrix_to_rows(W_K),
        "W_V"     => matrix_to_rows(W_V),
        "W_O"     => matrix_to_rows(W_O),
        "W_ff1"   => matrix_to_rows(W_ff1),
        "b_ff1"   => collect(b_ff1),        # vecteur simple
        "W_ff2"   => matrix_to_rows(W_ff2),
        "b_ff2"   => collect(b_ff2),
        "W_out"   => matrix_to_rows(W_out),
        "b_out"   => collect(b_out)
    ),
    "dims of params" => Dict(
        "W_embed" => size(W_embed),
        "W_Q" => size(W_Q),
        "W_K" => size(W_K),
        "W_V" => size(W_V),
        "W_O" => size(W_O),
        "W_ff1" => size(W_ff1),
        "b_ff1" => size(b_ff1),
        "W_ff2" => size(W_ff2),
        "b_ff2" => size(b_ff2),
        "W_out" => size(W_out),
        "b_out" => size(b_out)
    ),
    "stoi" => stoi,
    "itos" => itos,
    "merges" => merges,
    "vocab" => collect(vocab)
)

# Save to JSON
output_file = joinpath(@__DIR__, "weights.json")
open(output_file, "w") do f
    JSON3.write(f, weights_export)
end

println("✓ Weights exported to: $output_file")
println("File size: $(round(filesize(output_file) / 1024 / 1024, digits=2)) MB")
