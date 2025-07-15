import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

class RegularizationComparison:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.ridge_model = None
        self.lasso_model = None

    def train_ridge(self):
        self.ridge_model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', SGDRegressor(
                max_iter=10000,
                learning_rate='adaptive',
                eta0=0.01,
                penalty='l2',
                alpha=0.01,
                random_state=42
            ))
        ])
        self.ridge_model.fit(self.X_train, self.y_train)

    def train_lasso(self):
        self.lasso_model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', SGDRegressor(
                max_iter=10000,
                learning_rate='adaptive',
                eta0=0.01,
                penalty='l1',
                alpha=0.01,
                random_state=42
            ))
        ])
        self.lasso_model.fit(self.X_train, self.y_train)

    def evaluate_models(self):
        # Ridge Evaluation
        y_pred_ridge = self.ridge_model.predict(self.X_test)
        ridge_mse = mean_squared_error(self.y_test, y_pred_ridge)
        ridge_r2 = r2_score(self.y_test, y_pred_ridge)
        ridge_coef = self.ridge_model.named_steps['regressor'].coef_

        print("Ridge (L2) Regularization")
        print(f"MSE: {ridge_mse:.4f}")
        print(f"R² Score: {ridge_r2:.4f}")
        print(f"Coefficients: {ridge_coef}")
        print()

        # Lasso Evaluation
        y_pred_lasso = self.lasso_model.predict(self.X_test)
        lasso_mse = mean_squared_error(self.y_test, y_pred_lasso)
        lasso_coef = self.lasso_model.named_steps['regressor'].coef_

        print("Lasso (L1) Regularization")
        print(f"MSE: {lasso_mse:.4f}")
        print(f"Coefficients: {lasso_coef}")

# Example usage:
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv("fifa_players.csv")
    df.drop(columns=["nationality_name"], inplace=True)  # Adjust as necessary
      # Handle missing values 
    
    X = df.drop(columns=["overall"])  # Replace with your target column
    y = df["overall"]

    model_comparator = RegularizationComparison(X, y)
    model_comparator.train_ridge()
    model_comparator.train_lasso()
    