import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score


class PCAModel:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.pipeline = None
        self.pca = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_prepare_data(self):
        self.df = pd.read_csv(self.csv_path)
        if 'nationality_name' in self.df.columns:
            self.df.drop('nationality_name', axis=1, inplace=True)

        X = self.df.select_dtypes(include=['number']).dropna()
        y = X['overall']
        X = X.drop('overall', axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        return X.shape[1]  # return number of features for n_components max limit

    def train_model(self, n_components=27):
        self.pca = PCA(n_components=n_components, random_state=42)
        reg = LinearRegression()

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', self.pca),
            ('reg', reg),
        ])

        self.pipeline.fit(self.X_train, self.y_train)
        joblib.dump(self.pipeline, 'fifa.pkl')

    def evaluate_model(self):
        y_pred = self.pipeline.predict(self.X_test)
        return r2_score(self.y_test, y_pred)

    def get_variance_plot(self, save_path='static/images/pca_variance.png'):
        explained = self.pca.explained_variance_ratio_
        var_df = pd.DataFrame({
            'Principal Component': np.arange(1, len(explained)+1),
            'Explained Variance Ratio': explained
        })

        plt.figure(figsize=(8, 5))
        plt.bar(var_df['Principal Component'],
                var_df['Explained Variance Ratio'],
                label='Individual')
        plt.xlabel('Principal Component Index')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Variance Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return var_df

    def load_model_and_predict(self, input_data):
        model = joblib.load('fifa.pkl')
        return model.predict(input_data)
    
    

if __name__ == "__main__":
    pca_model = PCAModel('fifa_players.csv')
    n_features = pca_model.load_and_prepare_data()
    pca_model.train_model(n_components=min(27, n_features))
    r2_score = pca_model.evaluate_model()
    print(f"R2 Score: {r2_score}")
    variance_df = pca_model.get_variance_plot()
    print(variance_df)  
