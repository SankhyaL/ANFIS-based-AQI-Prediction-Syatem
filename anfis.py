import numpy as np

class ANFIS:
    def __init__(self, n_inputs, n_mfs=6, epochs=200, lr=0.001, patience=20):
        self.n_inputs = n_inputs
        self.n_mfs    = n_mfs
        self.epochs   = epochs
        self.lr       = lr
        self.patience = patience

        # Initialize MF parameters
        self.centers = np.tile(np.linspace(-1, 1, n_mfs), (n_inputs, 1))
        self.widths  = np.ones((n_inputs, n_mfs)) * 0.5

        # Consequent parameters
        self.params = np.random.randn(n_mfs, n_inputs + 1) * 0.01

    # Gaussian Membership Function
    def gaussian_mf(self, x, c, sigma):
        return np.exp(-0.5 * ((x - c) / (sigma + 1e-8)) ** 2)

    def forward(self, X):
        n = X.shape[0]

        # Layer 1: Fuzzification
        mu = np.zeros((n, self.n_inputs, self.n_mfs))
        for i in range(self.n_inputs):
            for j in range(self.n_mfs):
                mu[:, i, j] = self.gaussian_mf(
                    X[:, i], self.centers[i, j], self.widths[i, j]
                )

        # Layer 2: Rule strength (improved aggregation)
        w = np.sum(mu, axis=1)   # shape (n, n_mfs)

        # Normalize
        w_norm = w / (w.sum(axis=1, keepdims=True) + 1e-8)

        # Layer 3: Consequent
        X_aug = np.hstack([X, np.ones((n, 1))])
        f     = X_aug @ self.params.T

        # Output
        y_hat = (w_norm * f).sum(axis=1)

        return y_hat, w_norm, mu

    def train(self, X_train, y_train, X_val, y_val):
        best_loss      = float('inf')
        best_params    = self.params.copy()
        best_centers   = self.centers.copy()
        best_widths    = self.widths.copy()
        patience_count = 0
        history        = []

        for epoch in range(self.epochs):

            # Forward
            y_hat, w_norm, mu = self.forward(X_train)
            loss = np.mean((y_hat - y_train) ** 2)

            # ----------- CONSEQUENT LEARNING (LSE) -----------
            X_aug = np.hstack([X_train, np.ones((len(X_train), 1))])

            for k in range(self.n_mfs):
                wk = w_norm[:, k]
                W  = np.diag(wk)

                A = X_aug
                try:
                    theta = np.linalg.pinv(A.T @ W @ A) @ (A.T @ W @ y_train)
                    self.params[k] = theta
                except:
                    pass  # fallback if singular

            # ----------- MF PARAM UPDATE (Centers + Widths) -----------
            for i in range(self.n_inputs):
                for j in range(self.n_mfs):

                    # Center gradient
                    grad_c = np.mean(
                        (y_hat - y_train) * (X_train[:, i] - self.centers[i, j])
                    )

                    # Width gradient
                    mu_ij = mu[:, i, j]
                    grad_w = np.mean(
                        (y_hat - y_train) * mu_ij * ((X_train[:, i] - self.centers[i, j])**2) / (self.widths[i, j] ** 3 + 1e-8)
                    )

                    # Update with clipping
                    self.centers[i, j] -= self.lr * np.clip(grad_c, -0.1, 0.1)
                    self.widths[i, j]  -= self.lr * np.clip(grad_w, -0.1, 0.1)

                    # Keep widths stable
                    self.widths[i, j] = np.clip(self.widths[i, j], 0.05, 2.0)

            # ----------- VALIDATION -----------
            val_hat, _, _ = self.forward(X_val)
            val_loss = np.mean((val_hat - y_val) ** 2)

            history.append({
                'epoch': epoch,
                'train_loss': loss,
                'val_loss': val_loss
            })

            # Save best model
            if val_loss < best_loss:
                best_loss    = val_loss
                best_params  = self.params.copy()
                best_centers = self.centers.copy()
                best_widths  = self.widths.copy()
                patience_count = 0
            else:
                patience_count += 1

            # Early stopping
            if patience_count >= self.patience:
                print(f"Early stopping at epoch {epoch} | Best Val Loss: {best_loss:.5f}")
                self.params  = best_params
                self.centers = best_centers
                self.widths  = best_widths
                break

            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {loss:.5f} | Val Loss: {val_loss:.5f}")

        return history