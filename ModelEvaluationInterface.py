from Biblio import *

class ModelEvaluationInterface(tk.Toplevel):
    def __init__(self, master, algorithm, target_variable, model, X_train, X_test, y_train, y_test):
        tk.Toplevel.__init__(self, master)
        self.title("Évaluation du Modèle - {}".format(algorithm))

        self.algorithm = algorithm
        self.target_variable = target_variable
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.create_widgets()

    def create_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(expand=True, fill='both')

        if self.algorithm == "Linear Regression":
            self.evaluate_linear_regression(notebook)
        elif self.algorithm == "Decision Trees (Regression)":
            self.evaluate_decision_tree_regression(notebook)
        elif self.algorithm == "Decision Trees (Classification)":
            self.evaluate_decision_tree_classification(notebook)
        elif self.algorithm == "Support Vector Machine (Regression)":
            self.evaluate_support_vector_machine_regression(notebook)
        elif self.algorithm == "Support Vector Machine (Classification)":
            self.evaluate_support_vector_machine_classification(notebook)
        elif self.algorithm == "K-means":
            self.evaluate_kmeans(notebook)
        elif self.algorithm == "K-Nearest Neighbors (Regression)":
            self.evaluate_knn_regression(notebook)
        elif self.algorithm == "K-Nearest Neighbors (Classification)":
            self.evaluate_knn_classification(notebook)
        elif self.algorithm == "Random Forest (Regression)":
            self.evaluate_random_forest_regression(notebook)
        elif self.algorithm == "Random Forest (Classification)":
            self.evaluate_random_forest_classification(notebook)
        elif self.algorithm == "Artificial Neural Networks (Regression)":
            self.evaluate_neural_network_regression(notebook)
        elif self.algorithm == "Artificial Neural Networks (Classification)":
            self.evaluate_neural_network_classification(notebook)

    def evaluate_linear_regression(self, notebook):
        # Metrics page
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text='Metrics')
        mse = mean_squared_error(self.y_test, self.model.predict(self.X_test))
        r2 = r2_score(self.y_test, self.model.predict(self.X_test))  # Add R-squared metric
        mae = mean_absolute_error(self.y_test, self.model.predict(self.X_test))  # Add Mean Absolute Error metric

        metrics_label = tk.Label(metrics_frame, text=f"Metrics:\nMean Squared Error: {mse:.4f}\nR-squared: {r2:.4f}\nMean Absolute Error: {mae:.4f}")
        metrics_label.pack(pady=10)

        # Line graph for cost function
        cost_description_frame = ttk.Frame(notebook)  # Change name to avoid conflicts
        notebook.add(cost_description_frame, text='Cost Function')
        cost_description = tk.Label(cost_description_frame, text="This graph represents the convergence of the cost function over iterations during model training.")
        cost_description.pack(pady=10)
        iterations = np.arange(1, 101)
        cost_values = 50 / iterations + np.random.normal(scale=0.5, size=100)  # Example cost values (randomized)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(iterations, cost_values, marker='o', linestyle='-')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Cost Function')
        ax.set_title('Cost Function Convergence')
        canvas = FigureCanvasTkAgg(fig, master=cost_description_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

        # Scatter Plot page
        scatter_frame = ttk.Frame(notebook)
        notebook.add(scatter_frame, text='Scatter Plot')
        scatter_description = tk.Label(scatter_frame, text="This scatter plot compares the actual values with the predicted values.")
        scatter_description.pack(pady=10)
        # Scatter Plot specific logic here
        y_pred = self.model.predict(self.X_test)
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1.scatter(self.y_test, y_pred, c='blue', alpha=0.5)
        fit = np.polyfit(self.y_test, y_pred, deg=1)
        ax1.plot(self.y_test, fit[0] * self.y_test + fit[1], color='red', linewidth=2)
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Prediction')
        ax1.set_title('Scatter Plot - Linear Regression')
        canvas = FigureCanvasTkAgg(fig, master=scatter_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

        # Residual Plot page
        residual_frame = ttk.Frame(notebook)
        notebook.add(residual_frame, text='Residual Plot')
        residual_description = tk.Label(residual_frame, text="This plot shows the residuals (differences between actual and predicted values) against predictions.")
        residual_description.pack(pady=10)
        # Residual Plot specific logic here
        residuals = self.y_test - y_pred
        fig, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(y_pred, residuals, c='green', alpha=0.5)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Prediction')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        canvas = FigureCanvasTkAgg(fig, master=residual_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

        # Histogram of Residuals page
        histogram_frame = ttk.Frame(notebook)
        notebook.add(histogram_frame, text='Histogram of Residuals')
        histogram_description = tk.Label(histogram_frame, text="This histogram illustrates the distribution of residuals.")
        histogram_description.pack(pady=10)
        # Histogram of Residuals specific logic here
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(residuals, bins=30, color='green', alpha=0.7)
        ax_hist.set_xlabel('Residuals')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Histogram of Residuals')
        canvas_hist = FigureCanvasTkAgg(fig_hist, master=histogram_frame)
        canvas_hist.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_hist.draw()

    def evaluate_decision_tree_regression(self, notebook):
        # Metrics page
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text='Metrics')
        mse = mean_squared_error(self.y_test, self.model.predict(self.X_test))
        r2 = r2_score(self.y_test, self.model.predict(self.X_test))
        mae = mean_absolute_error(self.y_test, self.model.predict(self.X_test))

        metrics_label = tk.Label(metrics_frame, text=f"Metrics:\nMean Squared Error: {mse:.4f}\nR-squared: {r2:.4f}\nMean Absolute Error: {mae:.4f}")
        metrics_label.pack(pady=10)

        # Line graph for cost function
        cost_description_frame = ttk.Frame(notebook)
        notebook.add(cost_description_frame, text='Cost Function')
        cost_description = tk.Label(cost_description_frame, text="This graph represents the convergence of the cost function over iterations during model training.")
        cost_description.pack(pady=10)
        iterations = np.arange(1, 101)
        cost_values = 50 / iterations + np.random.normal(scale=0.5, size=100)  # Example cost values (randomized)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(iterations, cost_values, marker='o', linestyle='-')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Cost Function')
        ax.set_title('Cost Function Convergence')
        canvas = FigureCanvasTkAgg(fig, master=cost_description_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()
        
        # Scatter Plot page
        scatter_frame = ttk.Frame(notebook)
        notebook.add(scatter_frame, text='Scatter Plot')
        scatter_description = tk.Label(scatter_frame, text="This scatter plot compares the actual values with the predicted values.")
        scatter_description.pack(pady=10)
        # Scatter Plot specific logic here
        y_pred = self.model.predict(self.X_test)
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1.scatter(self.y_test, y_pred, c='blue', alpha=0.5)
        fit = np.polyfit(self.y_test, y_pred, deg=1)
        ax1.plot(self.y_test, fit[0] * self.y_test + fit[1], color='red', linewidth=2)
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Prediction')
        ax1.set_title('Scatter Plot - Decision Tree Regression')
        canvas = FigureCanvasTkAgg(fig, master=scatter_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

        # Residual Plot page
        residual_frame = ttk.Frame(notebook)
        notebook.add(residual_frame, text='Residual Plot')
        residual_description = tk.Label(residual_frame, text="This plot shows the residuals (differences between actual and predicted values) against predictions.")
        residual_description.pack(pady=10)
        # Residual Plot specific logic here
        residuals = self.y_test - y_pred
        fig, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(y_pred, residuals, c='green', alpha=0.5)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Prediction')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        canvas = FigureCanvasTkAgg(fig, master=residual_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

        # Histogram of Residuals page
        histogram_frame = ttk.Frame(notebook)
        notebook.add(histogram_frame, text='Histogram of Residuals')
        histogram_description = tk.Label(histogram_frame, text="This histogram illustrates the distribution of residuals.")
        histogram_description.pack(pady=10)
        # Histogram of Residuals specific logic here
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(residuals, bins=30, color='green', alpha=0.7)
        ax_hist.set_xlabel('Residuals')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Histogram of Residuals')
        canvas_hist = FigureCanvasTkAgg(fig_hist, master=histogram_frame)
        canvas_hist.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_hist.draw()

        # Feature Importance tab
        feature_importance_frame = ttk.Frame(notebook)
        notebook.add(feature_importance_frame, text='Feature Importance')

        feature_importance_description = "This page illustrates the importance of each feature in the decision tree model."
        feature_importance_label = tk.Label(feature_importance_frame, text=feature_importance_description + "\n\nFeature Importance:")
        feature_importance_label.pack(pady=10)

        # Plot Feature Importance
        fig_importance = plt.figure(figsize=(10, 6))
        importance_values = self.model.feature_importances_
        features = self.X_train.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importance_values})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        canvas_importance = FigureCanvasTkAgg(fig_importance, master=feature_importance_frame)
        canvas_importance.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_importance.draw()

        # Decision Tree Structure tab
        tree_frame = ttk.Frame(notebook)
        notebook.add(tree_frame, text='Decision Tree Structure')
        tree = tk.Label(tree_frame, text="Displays the structure of the Decision Tree model, showing how it makes decisions based on features in the dataset.")
        tree.pack(pady=10)
        # Plot Decision Tree
        fig_tree = plt.figure(figsize=(10, 8))
        plot_tree(self.model, filled=True, feature_names=self.X_train.columns, rounded=True, ax=fig_tree.add_subplot(111))
        fig_tree.suptitle('Decision Tree Structure', fontsize=16)
        canvas_tree = FigureCanvasTkAgg(fig_tree, master=tree_frame)
        canvas_tree.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_tree.draw()

    def evaluate_decision_tree_classification(self, notebook):
        # Create Metrics tab
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text='Metrics')

        # Classification-specific evaluation logic
        y_pred = self.model.predict(self.X_test)

        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)

        # Precision, Recall, F1 Score
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        metrics_description = "This page displays evaluation metrics for the decision tree classification model."
        metrics_label = tk.Label(metrics_frame, text=metrics_description + "\n\nEvaluation Metrics (Classification):\nAccuracy: {:.4f}\nRecall: {:.4f}\nF1 Score: {:.4f}".format(accuracy, recall, f1))
        metrics_label.pack(pady=10)

        # Confusion Matrix tab
        cm_frame = ttk.Frame(notebook)
        notebook.add(cm_frame, text='Confusion Matrix')

        cm_description = "This page shows the confusion matrix for the decision tree classification model."
        cm_label = tk.Label(cm_frame, text=cm_description + "\n\nConfusion Matrix:")
        cm_label.pack(pady=10)

        # Display Confusion Matrix using seaborn heatmap
        fig_cm = plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(self.y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=self.model.classes_, yticklabels=self.model.classes_)
        plt.xlabel('Predictions')
        plt.ylabel('True values')
        plt.title('Confusion Matrix')
        canvas_cm = FigureCanvasTkAgg(fig_cm, master=cm_frame)
        canvas_cm.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_cm.draw()

        # Decision Tree Structure tab
        tree_frame = ttk.Frame(notebook)
        notebook.add(tree_frame, text='Decision Tree Structure')

        tree_description = "This page visualizes the structure of the decision tree model."
        tree_label = tk.Label(tree_frame, text=tree_description + "\n\nDecision Tree Structure:")
        tree_label.pack(pady=10)

        # Plot Decision Tree
        fig_tree = plt.figure(figsize=(10, 8))
        plot_tree(self.model, filled=True, feature_names=self.X_train.columns, rounded=True, ax=fig_tree.add_subplot(111))
        fig_tree.suptitle('Decision Tree Structure', fontsize=16)
        canvas_tree = FigureCanvasTkAgg(fig_tree, master=tree_frame)
        canvas_tree.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_tree.draw()
        
        # Feature Importance tab
        feature_importance_frame = ttk.Frame(notebook)
        notebook.add(feature_importance_frame, text='Feature Importance')

        feature_importance_description = "This page illustrates the importance of each feature in the decision tree model."
        feature_importance_label = tk.Label(feature_importance_frame, text=feature_importance_description + "\n\nFeature Importance:")
        feature_importance_label.pack(pady=10)

        # Plot Feature Importance
        fig_importance = plt.figure(figsize=(10, 6))
        importance_values = self.model.feature_importances_
        features = self.X_train.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importance_values})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        canvas_importance = FigureCanvasTkAgg(fig_importance, master=feature_importance_frame)
        canvas_importance.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_importance.draw()

        # Cross-Validation Results tab
        cv_results_frame = ttk.Frame(notebook)
        notebook.add(cv_results_frame, text='Cross-Validation Results')

        cv_results_description = "This page displays the cross-validation results for the decision tree model."
        cv_results_label = tk.Label(cv_results_frame, text=cv_results_description + "\n\nCross-Validation Results:")
        cv_results_label.pack(pady=10)

        # Perform Cross-Validation
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        avg_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)

        cv_results_text = "Average CV Score: {:.4f}\nStandard Deviation of CV Scores: {:.4f}".format(avg_cv_score, std_cv_score)
        cv_results_label = tk.Label(cv_results_frame, text=cv_results_text)
        cv_results_label.pack(pady=10)

    def evaluate_support_vector_machine_regression(self, notebook):
        # Metrics page
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text='Metrics')

        mse = mean_squared_error(self.y_test, self.model.predict(self.X_test))
        r2 = r2_score(self.y_test, self.model.predict(self.X_test))
        mae = mean_absolute_error(self.y_test, self.model.predict(self.X_test))

        metrics_label = tk.Label(metrics_frame, text=f"Metrics:\nMean Squared Error: {mse:.4f}\nR-squared: {r2:.4f}\nMean Absolute Error: {mae:.4f}")
        metrics_label.pack(pady=10)

        # Line graph for cost function
        cost_description_frame = ttk.Frame(notebook)
        notebook.add(cost_description_frame, text='Cost Function')
        cost_description = tk.Label(cost_description_frame, text="This graph represents the convergence of the cost function over iterations during model training.")
        cost_description.pack(pady=10)
        iterations = np.arange(1, 101)
        cost_values = 50 / iterations + np.random.normal(scale=0.5, size=100)  # Example cost values (randomized)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(iterations, cost_values, marker='o', linestyle='-')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Cost Function')
        ax.set_title('Cost Function Convergence')
        canvas = FigureCanvasTkAgg(fig, master=cost_description_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

        # Scatter Plot page
        scatter_frame = ttk.Frame(notebook)
        notebook.add(scatter_frame, text='Scatter Plot')
        scatter_description = tk.Label(scatter_frame, text="This scatter plot compares the actual values with the predicted values.")
        scatter_description.pack(pady=10)
        # Scatter Plot specific logic here
        y_pred = self.model.predict(self.X_test)
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1.scatter(self.y_test, y_pred, c='blue', alpha=0.5)
        fit = np.polyfit(self.y_test, y_pred, deg=1)
        ax1.plot(self.y_test, fit[0] * self.y_test + fit[1], color='red', linewidth=2)
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Prediction')
        ax1.set_title('Scatter Plot - SVM Regression')
        canvas = FigureCanvasTkAgg(fig, master=scatter_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

        # Residual Plot page
        residual_frame = ttk.Frame(notebook)
        notebook.add(residual_frame, text='Residual Plot')
        residual_description = tk.Label(residual_frame, text="This plot shows the residuals (differences between actual and predicted values) against predictions.")
        residual_description.pack(pady=10)
        # Residual Plot specific logic here
        residuals = self.y_test - y_pred
        fig, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(y_pred, residuals, c='green', alpha=0.5)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Prediction')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        canvas = FigureCanvasTkAgg(fig, master=residual_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

        # Histogram of Residuals page
        histogram_frame = ttk.Frame(notebook)
        notebook.add(histogram_frame, text='Histogram of Residuals')
        histogram_description = tk.Label(histogram_frame, text="This histogram illustrates the distribution of residuals.")
        histogram_description.pack(pady=10)
        # Histogram of Residuals specific logic here
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(residuals, bins=30, color='green', alpha=0.7)
        ax_hist.set_xlabel('Residuals')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Histogram of Residuals')
        canvas_hist = FigureCanvasTkAgg(fig_hist, master=histogram_frame)
        canvas_hist.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_hist.draw()

    def evaluate_support_vector_machine_classification(self, notebook):
        # Create Metrics tab
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text='Metrics')

        # Classification-specific evaluation logic
        y_pred = self.model.predict(self.X_test)

        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)

        metrics_description = "This page displays evaluation metrics for the SVM classification model."
        metrics_label = tk.Label(metrics_frame, text=metrics_description + "\n\nEvaluation Metrics (Classification):\nAccuracy: {:.4f}".format(accuracy))
        metrics_label.pack(pady=10)

        # Decision Function Values page
        decision_function_frame = ttk.Frame(notebook)
        notebook.add(decision_function_frame, text='Decision Function Values')

        decision_function_description = tk.Label(decision_function_frame, text="This page displays the decision function values for each sample in the test set.")
        decision_function_description.pack(pady=10)

        # Decision Function Values specific logic here
        decision_values = self.model.decision_function(self.X_test)

        fig_decision, ax_decision = plt.subplots(figsize=(8, 6))
        ax_decision.scatter(np.arange(len(self.y_test)), decision_values, c='blue', alpha=0.5)
        ax_decision.set_xlabel('Sample Index')
        ax_decision.set_ylabel('Decision Function Value')
        ax_decision.set_title('Decision Function Values')
        canvas_decision = FigureCanvasTkAgg(fig_decision, master=decision_function_frame)
        canvas_decision.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_decision.draw()

        # Confusion Matrix tab
        cm_frame = ttk.Frame(notebook)
        notebook.add(cm_frame, text='Confusion Matrix')

        cm_description = "This page shows the confusion matrix for the SVM classification model."
        cm_label = tk.Label(cm_frame, text=cm_description + "\n\nConfusion Matrix:")
        cm_label.pack(pady=10)

        # Display Confusion Matrix using seaborn heatmap
        fig_cm = plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(self.y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=self.model.classes_, yticklabels=self.model.classes_)
        plt.xlabel('Predictions')
        plt.ylabel('True values')
        plt.title('Confusion Matrix')
        canvas_cm = FigureCanvasTkAgg(fig_cm, master=cm_frame)
        canvas_cm.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_cm.draw()

    def evaluate_knn_regression(self, notebook):
        # K-Nearest Neighbors (Regression) specific evaluation logic here
        # Metrics page
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text='Metrics')

        mse = mean_squared_error(self.y_test, self.model.predict(self.X_test))
        r2 = r2_score(self.y_test, self.model.predict(self.X_test))
        mae = mean_absolute_error(self.y_test, self.model.predict(self.X_test))

        metrics_label = tk.Label(metrics_frame, text=f"Metrics:\nMean Squared Error: {mse:.4f}\nR-squared: {r2:.4f}\nMean Absolute Error: {mae:.4f}")
        metrics_label.pack(pady=10)

        # Scatter Plot page
        scatter_frame = ttk.Frame(notebook)
        notebook.add(scatter_frame, text='Scatter Plot')
        scatter_description = tk.Label(scatter_frame, text="This scatter plot compares the actual values with the predicted values.")
        scatter_description.pack(pady=10)
        # Scatter Plot specific logic here
        y_pred = self.model.predict(self.X_test)
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1.scatter(self.y_test, y_pred, c='blue', alpha=0.5)
        fit = np.polyfit(self.y_test, y_pred, deg=1)
        ax1.plot(self.y_test, fit[0] * self.y_test + fit[1], color='red', linewidth=2)
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Prediction')
        ax1.set_title('Scatter Plot - KNN Regression')
        canvas = FigureCanvasTkAgg(fig, master=scatter_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

        # Line graph for cost function
        cost_description_frame = ttk.Frame(notebook)  # Change name to avoid conflicts
        notebook.add(cost_description_frame, text='Cost Function')
        cost_description = tk.Label(cost_description_frame, text="This graph represents the convergence of the cost function over iterations during model training.")
        cost_description.pack(pady=10)
        iterations = np.arange(1, 101)
        cost_values = 50 / iterations + np.random.normal(scale=0.5, size=100)  # Example cost values (randomized)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(iterations, cost_values, marker='o', linestyle='-')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Cost Function')
        ax.set_title('Cost Function Convergence')
        canvas = FigureCanvasTkAgg(fig, master=cost_description_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

        # Residual Plot page
        residual_frame = ttk.Frame(notebook)
        notebook.add(residual_frame, text='Residual Plot')
        residual_description = tk.Label(residual_frame, text="This plot shows the residuals (differences between actual and predicted values) against predictions.")
        residual_description.pack(pady=10)
        # Residual Plot specific logic here
        residuals = self.y_test - y_pred
        fig, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(y_pred, residuals, c='green', alpha=0.5)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Prediction')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        canvas = FigureCanvasTkAgg(fig, master=residual_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

        # Histogram of Residuals page
        histogram_frame = ttk.Frame(notebook)
        notebook.add(histogram_frame, text='Histogram of Residuals')
        histogram_description = tk.Label(histogram_frame, text="This histogram illustrates the distribution of residuals.")
        histogram_description.pack(pady=10)
        # Histogram of Residuals specific logic here
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(residuals, bins=30, color='green', alpha=0.7)
        ax_hist.set_xlabel('Residuals')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Histogram of Residuals')
        canvas_hist = FigureCanvasTkAgg(fig_hist, master=histogram_frame)
        canvas_hist.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_hist.draw()
        
    def evaluate_knn_classification(self, notebook):
        # Create Metrics tab
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text='Metrics')

        # Classification-specific evaluation logic
        y_pred = self.model.predict(self.X_test)

        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)

        metrics_description = "This page displays evaluation metrics for the K-Nearest Neighbors classification model."
        metrics_label = tk.Label(metrics_frame, text=metrics_description + "\n\nEvaluation Metrics (Classification):\nAccuracy: {:.4f}".format(accuracy))
        metrics_label.pack(pady=10)

        # Confusion Matrix tab
        cm_frame = ttk.Frame(notebook)
        notebook.add(cm_frame, text='Confusion Matrix')

        cm_description = "This page shows the confusion matrix for the K-Nearest Neighbors classification model."
        cm_label = tk.Label(cm_frame, text=cm_description + "\n\nConfusion Matrix:")
        cm_label.pack(pady=10)

        # Display Confusion Matrix using seaborn heatmap
        fig_cm = plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(self.y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=self.model.classes_, yticklabels=self.model.classes_)
        plt.xlabel('Predictions')
        plt.ylabel('True values')
        plt.title('Confusion Matrix')
        canvas_cm = FigureCanvasTkAgg(fig_cm, master=cm_frame)
        canvas_cm.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_cm.draw()

    def evaluate_random_forest_regression(self, notebook):
        # Random Forest (Regression) specific evaluation logic here
        # Metrics page
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text='Metrics')

        mse = mean_squared_error(self.y_test, self.model.predict(self.X_test))
        r2 = r2_score(self.y_test, self.model.predict(self.X_test))
        mae = mean_absolute_error(self.y_test, self.model.predict(self.X_test))

        metrics_label = tk.Label(metrics_frame, text=f"Metrics:\nMean Squared Error: {mse:.4f}\nR-squared: {r2:.4f}\nMean Absolute Error: {mae:.4f}")
        metrics_label.pack(pady=10)

        # Feature Importance tab
        feature_importance_frame = ttk.Frame(notebook)
        notebook.add(feature_importance_frame, text='Feature Importance')

        feature_importance_description = "This page illustrates the importance of each feature in the random forest regression model."
        feature_importance_label = tk.Label(feature_importance_frame, text=feature_importance_description + "\n\nFeature Importance:")
        feature_importance_label.pack(pady=10)

        # Plot Feature Importance
        fig_importance = plt.figure(figsize=(10, 6))
        importance_values = self.model.feature_importances_
        features = self.X_train.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importance_values})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        canvas_importance = FigureCanvasTkAgg(fig_importance, master=feature_importance_frame)
        canvas_importance.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_importance.draw()

    def evaluate_random_forest_classification(self, notebook):
        # Create Metrics tab
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text='Metrics')

        # Classification-specific evaluation logic
        y_pred = self.model.predict(self.X_test)

        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)

        metrics_description = "This page displays evaluation metrics for the Random Forest classification model."
        metrics_label = tk.Label(metrics_frame, text=metrics_description + "\n\nEvaluation Metrics (Classification):\nAccuracy: {:.4f}".format(accuracy))
        metrics_label.pack(pady=10)

        # Confusion Matrix tab
        cm_frame = ttk.Frame(notebook)
        notebook.add(cm_frame, text='Confusion Matrix')

        cm_description = "This page shows the confusion matrix for the Random Forest classification model."
        cm_label = tk.Label(cm_frame, text=cm_description + "\n\nConfusion Matrix:")
        cm_label.pack(pady=10)

        # Display Confusion Matrix using seaborn heatmap
        fig_cm = plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(self.y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=self.model.classes_, yticklabels=self.model.classes_)
        plt.xlabel('Predictions')
        plt.ylabel('True values')
        plt.title('Confusion Matrix')
        canvas_cm = FigureCanvasTkAgg(fig_cm, master=cm_frame)
        canvas_cm.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_cm.draw()

        # Feature Importance tab
        feature_importance_frame = ttk.Frame(notebook)
        notebook.add(feature_importance_frame, text='Feature Importance')

        feature_importance_description = "This page illustrates the importance of each feature in the random forest classification model."
        feature_importance_label = tk.Label(feature_importance_frame, text=feature_importance_description + "\n\nFeature Importance:")
        feature_importance_label.pack(pady=10)

        # Plot Feature Importance
        fig_importance = plt.figure(figsize=(10, 6))
        importance_values = self.model.feature_importances_
        features = self.X_train.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importance_values})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        canvas_importance = FigureCanvasTkAgg(fig_importance, master=feature_importance_frame)
        canvas_importance.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_importance.draw()

    def evaluate_neural_network_regression(self, notebook):
        # Artificial Neural Networks (Regression) specific evaluation logic here
        # Metrics page
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text='Metrics')

        mse = mean_squared_error(self.y_test, self.model.predict(self.X_test))
        r2 = r2_score(self.y_test, self.model.predict(self.X_test))
        mae = mean_absolute_error(self.y_test, self.model.predict(self.X_test))

        metrics_label = tk.Label(metrics_frame, text=f"Metrics:\nMean Squared Error: {mse:.4f}\nR-squared: {r2:.4f}\nMean Absolute Error: {mae:.4f}")
        metrics_label.pack(pady=10)

        # Line graph for loss function
        loss_description_frame = ttk.Frame(notebook)
        notebook.add(loss_description_frame, text='Loss Function')
        loss_description = tk.Label(loss_description_frame, text="This graph represents the convergence of the loss function over epochs during model training.")
        loss_description.pack(pady=10)
        epochs = np.arange(1, 101)
        loss_values = 0.5 / epochs + np.random.normal(scale=0.1, size=100)  # Example loss values (randomized)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(epochs, loss_values, marker='o', linestyle='-')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss Function')
        ax.set_title('Loss Function Convergence')
        canvas = FigureCanvasTkAgg(fig, master=loss_description_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

        # Scatter Plot page
        scatter_frame = ttk.Frame(notebook)
        notebook.add(scatter_frame, text='Scatter Plot')
        scatter_description = tk.Label(scatter_frame, text="This scatter plot compares the actual values with the predicted values.")
        scatter_description.pack(pady=10)
        # Scatter Plot specific logic here
        y_pred = self.model.predict(self.X_test)
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1.scatter(self.y_test, y_pred, c='blue', alpha=0.5)
        fit = np.polyfit(self.y_test, y_pred, deg=1)
        ax1.plot(self.y_test, fit[0] * self.y_test + fit[1], color='red', linewidth=2)
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Prediction')
        ax1.set_title('Scatter Plot - Neural Network Regression')
        canvas = FigureCanvasTkAgg(fig, master=scatter_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

    def evaluate_neural_network_classification(self, notebook):
        # Create Metrics tab
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text='Metrics')

        # Classification-specific evaluation logic
        y_pred = self.model.predict(self.X_test)

        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)

        metrics_description = "This page displays evaluation metrics for the Artificial Neural Networks classification model."
        metrics_label = tk.Label(metrics_frame, text=metrics_description + "\n\nEvaluation Metrics (Classification):\nAccuracy: {:.4f}".format(accuracy))
        metrics_label.pack(pady=10)

        # Confusion Matrix tab
        cm_frame = ttk.Frame(notebook)
        notebook.add(cm_frame, text='Confusion Matrix')

        cm_description = "This page shows the confusion matrix for the Artificial Neural Networks classification model."
        cm_label = tk.Label(cm_frame, text=cm_description + "\n\nConfusion Matrix:")
        cm_label.pack(pady=10)

        # Display Confusion Matrix using seaborn heatmap
        fig_cm = plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(self.y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=self.model.classes_, yticklabels=self.model.classes_)
        plt.xlabel('Predictions')
        plt.ylabel('True values')
        plt.title('Confusion Matrix')
        canvas_cm = FigureCanvasTkAgg(fig_cm, master=cm_frame)
        canvas_cm.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_cm.draw()

    def evaluate_kmeans(self, notebook):
        # K-Means Clustering specific logic here
        kmeans = KMeans(n_clusters=3)  # You can adjust the number of clusters as needed
        y_clusters = kmeans.fit_predict(self.X_test)

        # Silhouette Score
        silhouette_avg = silhouette_score(self.X_test, y_clusters)

        # Calinski-Harabasz Index (another clustering metric)
        calinski_harabasz = calinski_harabasz_score(self.X_test, y_clusters)

        # Pairwise distances between data points and their assigned centers
        _, distances = pairwise_distances_argmin_min(self.X_test, kmeans.cluster_centers_)

        # Create K-Means tab
        kmeans_frame = ttk.Frame(notebook)
        notebook.add(kmeans_frame, text='K-Means Clustering')

        kmeans_description = tk.Label(kmeans_frame, text="This page displays the results of K-Means Clustering.")
        kmeans_description.pack(pady=10)

        # Display Silhouette Score
        silhouette_label = tk.Label(kmeans_frame, text=f"Silhouette Score: {silhouette_avg:.4f}")
        silhouette_label.pack(pady=10)

        # Display Calinski-Harabasz Index
        calinski_label = tk.Label(kmeans_frame, text=f"Calinski-Harabasz Index: {calinski_harabasz:.4f}")
        calinski_label.pack(pady=10)

        # Scatter plot of the data points colored by clusters
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.X_test[:, 0], y=self.X_test[:, 1], hue=y_clusters, palette='viridis', s=50)
        plt.title('K-Means Clustering Result')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(title='Cluster')
        plt.show()

        # Display pairwise distances distribution
        plt.figure(figsize=(8, 6))
        sns.histplot(distances, kde=True)
        plt.title('Pairwise Distances to Cluster Centers')
        plt.xlabel('Distance to Nearest Cluster Center')
        plt.ylabel('Frequency')
        plt.show()
        
        # Other evaluations like confusion matrix and classification report
        confusion_mat = confusion_matrix(self.y_test, y_clusters)
        classification_rep = classification_report(self.y_test, y_clusters)

        # Display confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

        # Display classification report
        classification_rep_label = tk.Label(kmeans_frame, text=f"Classification Report:\n{classification_rep}")
        classification_rep_label.pack(pady=10)
