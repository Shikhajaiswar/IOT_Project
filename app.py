"""
__author__ = "Vivekanand Mishra"
__copyright__ = "Copyright 2023"
__email__ = "mvivekanandji@gmail.com"
__status__ = "Development"

"""
import streamlit as st #for rapid dev
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

#
@st.cache
def read_data(file) -> pd.DataFrame:
    # Read the file content using pandas
    return pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)


# # Load the data
# @st.cache
# def load_data():
#     df = pd.read_csv('test.csv')
#     return df


def main():
    app_name = 'Predictory Analysis 📈 '
    st.set_page_config(page_title=app_name)
    st.title(app_name)

    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload data file", type=["xlsx", "csv"])

    if uploaded_file is not None:
        df = read_data(uploaded_file)

        with st.expander("Data", expanded=True):
            st.table(df.head(10))

        with st.expander("Data Stats", expanded=True):
            st.table(df.describe())

        with st.sidebar.expander("Configs"):
            features = ['TEMPERATURE', 'HUMIDITY', 'VIBRATION MAGNITUDE']
            independent_variables = st.multiselect('Select independent variables',
                                                           options=df.columns, default=features[:-1])
            dependent_variable = [feature for feature in features if feature not in independent_variables]
            error_threshold = st.number_input('Error threshold', value=15, min_value=0, max_value=100)

            if len(independent_variables) > 2:
                st.sidebar.error("Please select a maximum of two independent variables.")

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "KNN": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(),
                "SVR": SVR()
            }
            selected_models = st.multiselect('Select models to use', list(models.keys()), default=list(models.keys()))
            models_to_use = {name: models[name] for name in selected_models}

        independent_var_1 = st.sidebar.number_input(f'Enter {independent_variables[0].capitalize()}', min_value=0)
        independent_var_2 = st.sidebar.number_input(f'Enter {independent_variables[1].capitalize()}', min_value=0,
                                                    max_value=100)
        predict = st.sidebar.button("Predict")

        if predict:
            st.markdown("""---""")
            st.markdown("""---""")
            st.subheader("Comparison of Model Performance: Mean Squared Error (MSE)")
            # X = df[['TEMPERATURE', 'HUMIDITY']]
            # y = df['VIBRATION MAGNITUDE']
            X = df[independent_variables]
            y = df[dependent_variable]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Feature Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Model Training
            for name, model in models_to_use.items():
                model.fit(X_train_scaled, y_train)

            # Model Evaluation and Plotting
            plt.figure(figsize=(12, 8))

            results_df = pd.DataFrame(columns=['Model', 'Mean Squared Error'])

            #finding MSE
            for name, model in models_to_use.items():
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                # results_df = results_df.append({'Model': name, 'Mean Squared Error': mse}, ignore_index=True)
                new_row = {'Model': name, 'Mean Squared Error': mse}
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                plt.scatter(y_test, y_pred, label=name)

            # Display the DataFrame as a table
            st.table(results_df)

            plt.plot(y_test, y_test, color='red')
            plt.xlabel('Actual Vibration')
            plt.ylabel('Predicted Vibration')
            plt.title('Actual vs Predicted Vibration')
            plt.legend()
            st.pyplot()

            # Calculate the correlation matrix
            corr_matrix = X_train.corr()

            # Plot the heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Heatmap')
            st.pyplot()

            plt.figure(figsize=(10, 8))
            sns.barplot(x='Mean Squared Error', y='Model', data=results_df, palette='viridis')
            plt.xlabel('Mean Squared Error')
            plt.ylabel('Model')
            plt.title('Mean Squared Error for Each Model')
            plt.xlim(0, max(results_df['Mean Squared Error']) * 1.1)
            st.pyplot()

            scaled_input = scaler.transform([[independent_var_1, independent_var_2]])

            st.markdown("""---""")
            st.markdown("""---""")
            st.subheader("Individual models")
            for name, model in models_to_use.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"#### {name} (MSE:{mse})")
                predicted_vibration = model.predict(scaled_input)
                st.write(
                    f"Predicted {dependent_variable[0].capitalize()} with "
                    f"{independent_variables[0].capitalize()} as {independent_var_1} and "
                    f"{independent_variables[1].capitalize()} as {independent_var_2} is "
                    f"{predicted_vibration}")

                if predicted_vibration>error_threshold:
                    st.warning("Value crossed the threshold!")

                plt.figure(figsize=(8, 6))
                plt.scatter(y_test, y_pred)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                plt.xlabel('Actual Vibration')
                plt.ylabel('Predicted Vibration')
                plt.title(f'Actual vs Predicted Vibration - {name}')
                st.pyplot()
                st.markdown("""---""")


if __name__ == '__main__':
    main()
