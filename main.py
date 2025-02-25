# Define model, loss, optimizer, and data loaders
model = DeepRBFNetwork(feature_extractor, num_classes=2, feature_dim=128)
criterion = CustomLoss(lambda_margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define save directory
save_dir = "checkpoints"

# Initialize trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    save_dir=save_dir
)

# Train the model
trainer.train(num_epochs=20)