import streamlit as st


def intro_page():
    st.markdown(
        """
        <style>
        .title {
            font-size: 2.5em;
            font-weight: bold;
            color: #2C3E50;
            text-align: center;
            margin-bottom: 20px;
        }
        .section-header {
            font-size: 1.5em;
            font-weight: bold;
            color: #34495E;
            margin-top: 30px;
        }
        .subsection {
            font-size: 1.2em;
            font-weight: bold;
            color: #5D6D7E;
            margin-top: 20px;
        }
        .content {
            font-size: 1em;
            color: #566573;
            line-height: 1.6;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            font-size: 0.9em;
            color: #7F8C8D;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Welcome to the Budget Forecasting App")
    st.markdown(
        """
        <div class="content">
            This application offers a comprehensive suite of tools for data analysis, forecasting, and machine learning model training. Navigate through the different sections using the menu on the left to access various functionalities.<br>
            The suite specifically supports the analysis of cantonal data and the models are specially selected and prepared for two datasets.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-header">Features</div>', unsafe_allow_html=True)

    st.markdown('<div class="subsection">Data Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul class="content">
            <li><strong>Dataset Overview</strong>: Get a first glance at your dataset.</li>
            <li><strong>Interactive Data Exploration</strong>: Analyze your data interactively by selecting different columns for the x and y axes.</li>
            <li><strong>Percentage Difference Analysis</strong>: Visualize the percentage difference between Budget y and Realized values.</li>
            <li><strong>Heatmap of Percentage Differences</strong>: Generate a heatmap to compare percentage differences across different regions and years.</li>
            <li><strong>Custom Plots</strong>: Create custom plots based on region, ACC ID, and other variables of interest.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="subsection">Forecasting</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="content">
            The Forecasting section allows you to load your dataset and choose from five different forecasting models:
            <ol>
                <li><strong>XGBoost</strong></li>
                <li><strong>RandomForest</strong></li>
                <li><strong>CatBoost</strong></li>
                <li><strong>RNN</strong></li>
                <li><strong>Prophet</strong></li>
            </ol>
            After selecting a model, you can tune its hyperparameters. The app will then train the model and display:
            <ul>
                <li>The best hyperparameters found</li>
                <li>Model performance metrics</li>
                <li>Feature importance (for models that support it)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="subsection">Results</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="content">
            The results of the individual forecasting models can be viewed here. Select a region and Acc-ID to see the predictions of different models.
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    intro_page()
