from .losses import hybrid_loss_function


class Trainer:
    def __init__(self, model, optimizer, epochs=5000):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs

    def train_model(self, X_data, y_data, X_phys):
        self.model.train()

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            
            loss, l_data, l_phys, l_ic = hybrid_loss_function(self.model, X_data, y_data, X_phys)
            loss.backward()
            self.optimizer.step()

            if epoch % 500 == 0:
                print(f"Epoch {epoch:04d} | Total Loss: {loss.item():.6e} | "
                      f"Data: {l_data.item():.6e} | Phys: {l_phys.item():.6e}")

        return loss