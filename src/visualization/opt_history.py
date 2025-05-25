from history import log_data

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go # For interactive 3D plots if desired
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = "plotly_white" # Clean plotly theme


import re
import pandas as pd
import ast # For safely evaluating the parameters dictionary string



parsed_data = []
# Regex to capture trial number, value, parameters, and best trial info
# Corrected regex to handle missing "Best is trial..." line if it's the first trial
regex = re.compile(
    r"Trial\s+(?P<trial_num>\d+)\s+finished\s+with\s+value:\s+(?P<value>[\d.]+)\s+and\s+parameters:\s+(?P<params>\{.*?\})\.\s*(?:Best\s+is\s+trial\s+\d+\s+with\s+value:\s+(?P<best_value_so_far>[\d.]+)\.)?"
)

for line in log_data.strip().split('\n'):
    if "Trial" in line and "finished" in line:
        match = regex.search(line)
        if match:
            data_dict = match.groupdict()
            try:
                params_dict = ast.literal_eval(data_dict['params'])
                parsed_data.append({
                    'trial': int(data_dict['trial_num']),
                    'value': float(data_dict['value']),
                    'best_value_so_far': float(data_dict['best_value_so_far']) if data_dict.get('best_value_so_far') else float(data_dict['value']), # Handle first trial
                    **params_dict # Unpack parameters into the main dictionary
                })
            except Exception as e:
                print(f"Skipping line due to parsing error: {line}\nError: {e}")
        else:
            print(f"Regex did not match line: {line}")


df_trials = pd.DataFrame(parsed_data)
df_trials = df_trials.sort_values(by='trial').reset_index(drop=True)

# For the first trial, 'best_value_so_far' might be missing in the log or equal to its own value
# We can fill forward the 'best_value_so_far' or recalculate it
current_best = float('-inf')
best_values_progressive = []
for val in df_trials['value']:
    if val > current_best:
        current_best = val
    best_values_progressive.append(current_best)
df_trials['best_value_progressive'] = best_values_progressive


print(f"Parsed {len(df_trials)} trials into DataFrame.")
if not df_trials.empty:
    print(df_trials.head())
    # Identify hyperparameters (assuming they are all columns except 'trial', 'value', 'best_value_so_far', 'best_value_progressive')
    hyperparameters = [col for col in df_trials.columns if col not in ['trial', 'value', 'best_value_so_far', 'best_value_progressive']]
    print("\nIdentified Hyperparameters:", hyperparameters)
else:
    print("No trials parsed. Check the log format and regex.")
    hyperparameters = [] # Define as empty list to avoid NameError later

# --- Global Matplotlib Styling for Publication Quality (Light Theme) ---
sns.set_theme(style="whitegrid") # Base style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans'] # Professional fonts
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#333333' # Dark grey for axis lines
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['text.color'] = '#333333'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.dpi'] = 100 # for display, savefig will be higher


if not df_trials.empty:
    # --- 1. Optimization History Plot (Most Important) ---
    # Shows how the objective value improved over trials.
    plt.figure(figsize=(12, 7))
    plt.plot(df_trials['trial'], df_trials['value'], marker='o', linestyle='-', color='lightgrey', alpha=0.6, label='Objective Value per Trial', zorder=1)
    plt.plot(df_trials['trial'], df_trials['best_value_progressive'], marker='.', linestyle='-', color='crimson', linewidth=2.5, label='Best Value So Far', zorder=2)

    # Highlight the overall best trial
    best_trial_overall_idx = df_trials['value'].idxmax()
    best_trial_overall = df_trials.loc[best_trial_overall_idx]
    plt.scatter(best_trial_overall['trial'], best_trial_overall['value'],
                color='gold', s=200, edgecolor='black', zorder=3, label=f"Overall Best (Trial {best_trial_overall['trial']:.0f})")

    plt.xlabel("Trial Number")
    plt.ylabel("Objective Value (e.g., Accuracy)")
    plt.title("Hyperparameter Optimization History", fontweight='bold')
    plt.legend(frameon=True, loc='lower right', facecolor='white', framealpha=0.8)
    plt.grid(True, linestyle='--', alpha=0.7)
    sns.despine() # Cleaner look
    plt.tight_layout()
    plt.savefig("optuna_optimization_history.png", dpi=300)
    plt.show()

    # --- 2. Parallel Coordinate Plot for Hyperparameters ---
    # Shows relationships between hyperparameters and the objective value.
    # Requires Optuna to be installed, or we can build a similar one manually for selected params.
    # For manual: Normalize parameters for better visualization.
    if hyperparameters: # Check if hyperparameters were identified
        df_parallel = df_trials[['value'] + hyperparameters].copy()

        # Normalize hyperparameter columns for parallel coordinates
        # Log scale for 'learning_rate', 'gamma', 'lambda', 'alpha' might be better if they span orders of magnitude
        log_params = ['learning_rate', 'gamma', 'lambda', 'alpha']
        for p in log_params:
            if p in df_parallel.columns:
                 # Add a small epsilon to avoid log(0)
                df_parallel[p] = np.log10(df_parallel[p] + 1e-9) # 1e-9 is a small epsilon

        # Min-Max scale all parameter columns (including log-transformed ones)
        for col in hyperparameters: # Iterate over original hyperparameter names
            # Check if column exists (it might have been log-transformed)
            col_to_scale = col if col not in log_params else col # Use original name if not log-transformed
            if col_to_scale in df_parallel.columns:
                min_val = df_parallel[col_to_scale].min()
                max_val = df_parallel[col_to_scale].max()
                if max_val > min_val: # Avoid division by zero
                    df_parallel[col_to_scale] = (df_parallel[col_to_scale] - min_val) / (max_val - min_val)
                else:
                    df_parallel[col_to_scale] = 0.5 # Assign a neutral value if all values are the same

        # Add trial number for coloring by progression (optional)
        df_parallel['trial_group'] = pd.cut(df_trials['trial'], bins=5, labels=False) # Group trials for color

        plt.figure(figsize=(16, 8))
        # Using Plotly for a potentially more interactive parallel coordinates plot
        
        # Find the best trial
        best_trial_idx = df_trials['value'].idxmax()
        
        # Create dimensions for Plotly parallel coordinates
        dimensions = []
        # First, the objective value
        dimensions.append(dict(
            label='Objective Value', 
            values=df_trials['value'],
            # range=[df_trials['value'].min(), df_trials['value'].max()] # Optional: can be auto-detected
        ))
        # Then, the hyperparameters
        for param in hyperparameters:
            col_name = param
            label = f"log10({param})" if param in log_params and param in df_parallel.columns else param # Use original name if not log-transformed
            
            # Ensure we are using the potentially log-transformed data for actual values in parallel plot if it was scaled
            # However, the 'dimensions' in Parcoords usually take the original scale for display of ticks,
            # but values from the dataframe passed for plotting.
            # Let's use original df_trials values for the dimensions for clarity of ranges,
            # but ensure line colors are based on the actual objective.
            if param in df_trials.columns:
                 dimensions.append(dict(
                    label=label, 
                    values=df_trials[param],
                    # range=[df_trials[param].min(), df_trials[param].max()] # Optional
                ))

        if len(dimensions) > 1: # Need at least one param
            
            # --- Logic for coloring the best trial line ---
            # We'll create an array of colors. Most lines will be colored by their 'value'
            # via colorscale. The best trial will get a distinct color.
            # Plotly Parcoords line.color can accept an array mapping to each line.
            # However, to make one line red and others by colorscale is tricky.
            # A simpler approach for visual distinction within Parcoords limitations:
            # Add a new "artificial" dimension that helps us color the best line differently.
            # This is a workaround as Parcoords doesn't have a direct "highlight specific line" feature
            # with different style properties like width.
            
            # We will rely on the existing `line.color = df_trials['value']` and a good `colorscale`.
            # To make the best trial stand out more with color, we could try to manipulate
            # the colorscale or the values passed to `line.color` but it's non-trivial to make
            # just *one* line a specific color (e.g., bright red) while others follow a scale
            # without that red being part of the scale.

            # Given the constraints, the most straightforward way to highlight with Plotly Parcoords
            # is to ensure the colorscale used makes high values (best trial) very distinct.
            # We can't easily make it "thick red" directly.
            # The existing setup already colors by 'value'. We can choose a colorscale
            # where the top value (best trial) is a prominent color.
            # E.g., 'RdYlGn' (Red-Yellow-Green) or 'RdBu'. If higher is better, green/blue would be good.
            # If you want the best to be red, and it's the highest value, use a reversed scale like 'viridis_r'
            # if red is at the low end of viridis, or a scale like 'Reds' where higher values are darker red.

            # Let's try to ensure the best trial gets a very distinct color from the colorscale.
            # The current 'viridis' makes high values yellow-ish.
            # If 'crimson' is desired for best, and others on a blue/green scale:
            # This would require multiple Parcoords traces or more complex data manipulation.

            # Sticking to Plotly's standard way:
            # Color by value, and use a colorscale where the max value is distinct.
            # If you want the *best* to be red, and *best means highest value*,
            # then a colorscale like 'Reds' or 'Hot' might work where high values are red/hot.
            
            fig_par_coords = go.Figure(data=
                go.Parcoords(
                    line = dict(color = df_trials['value'], # Color lines by objective value
                               # Choose a colorscale where the max (best value) is prominent and reddish if desired
                               colorscale = 'Reds', # e.g., 'Reds', 'Hot', or 'RdYlGn' (adjust if lower is better)
                               showscale = True,
                               cmin = df_trials['value'].min(),
                               cmax = df_trials['value'].max()
                               # For line width, Parcoords does not support per-line width.
                               ),
                    dimensions = dimensions
                )
            )
            fig_par_coords.update_layout(
                title='Parallel Coordinates of Hyperparameters and Objective Value (Best Trial in Darker Red)',
                font=dict(size=10)
            )
            fig_par_coords.show()
            fig_par_coords.write_image("optuna_parallel_coordinates.png", scale=2) # Save as static image
        else:
            print("Not enough dimensions for parallel coordinates plot after processing.")


    # --- 3. Hyperparameter Importance Plot (More Advanced - needs Optuna study object or manual calculation) ---
    # If you have the Optuna 'study' object, you can use:
    # import optuna.visualization as vis
    # fig = vis.plot_param_importances(study)
    # fig.show() / fig.write_image("optuna_param_importances.png")
    # For manual, one simple way is to see correlation or use feature importance from a model predicting 'value'
    # Let's do a simplified version: Scatter plots of each hyperparameter vs. objective value

    if hyperparameters: # Check if hyperparameters were identified
        num_params = len(hyperparameters)
        # Determine grid size for subplots
        cols = 3
        rows = int(np.ceil(num_params / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5), sharey=False)
        axes = axes.flatten() # Flatten to 1D array for easy iteration

        for i, param in enumerate(hyperparameters):
            if i < len(axes): # Ensure we don't try to plot more than available subplots
                ax = axes[i]
                # Use a perceptually uniform colormap like 'viridis' or 'plasma'
                sc = ax.scatter(df_trials[param], df_trials['value'],
                                c=df_trials['trial'], cmap='viridis', alpha=0.7, s=50)
                ax.set_xlabel(param)
                ax.set_ylabel("Objective Value")
                ax.set_title(f"Value vs. {param}", fontsize=14)
                if param in log_params: # Indicate if x-axis is log-scaled in practice (though we plot original values here)
                    ax.set_xscale('log') # Apply log scale if appropriate for the parameter
                ax.grid(True, linestyle=':', alpha=0.5)

        # Add a colorbar for the trial number
        if num_params > 0: # only if there are plots
            fig.colorbar(sc, ax=axes[:num_params], orientation='horizontal', fraction=0.05, pad=0.1, label='Trial Number')

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle("Objective Value vs. Individual Hyperparameters", fontsize=20, fontweight='bold', y=1.03 if rows > 1 else 1.05)
        plt.tight_layout(rect=[0, 0, 1, 0.98 if rows > 1 else 0.95])
        plt.savefig("optuna_param_vs_value_scatter.png", dpi=300)
        plt.show()

    # --- 4. Slice Plot (Needs Optuna study object for direct plotting) ---
    # Shows slices of the objective function for selected parameters.
    # import optuna.visualization as vis
    # fig = vis.plot_slice(study, params=['n_estimators', 'learning_rate', 'max_depth']) # Choose key params
    # fig.show() / fig.write_image("optuna_slice_plot.png")
    # Manual alternative: Similar to scatter plots above, but could be 2D heatmaps for pairs if data is dense.

    # --- 5. Contour Plot for two most important parameters (if identifiable) ---
    # This is more advanced and usually requires the Optuna study object or a good way to determine importance.
    # For now, let's pick two common important ones like 'learning_rate' and 'n_estimators'.
    if 'learning_rate' in df_trials.columns and 'n_estimators' in df_trials.columns:
        plt.figure(figsize=(10, 8))
        # Using Plotly for a nicer interactive contour plot
        fig_contour = go.Figure(data =
            go.Contour(
                z=df_trials['value'],
                x=df_trials['learning_rate'],
                y=df_trials['n_estimators'],
                colorscale='viridis', # Choose a nice colorscale
                colorbar=dict(title='Objective Value'),
                contours=dict(
                    coloring='heatmap', # or 'lines' or 'fill'
                    showlabels=True, # Show labels on contour lines
                    labelfont=dict( # Font properties for contour labels
                        size=12,
                        color='white',
                    )
                ),
                # hovertemplate='LR: %{x}<br>N_Est: %{y}<br>Value: %{z}<extra></extra>'
            )
        )
        fig_contour.update_layout(
            title='Contour Plot: Objective Value vs. Learning Rate & N_Estimators',
            xaxis_title='Learning Rate (log scale)',
            yaxis_title='N_Estimators',
            xaxis_type='log' # Important for learning rate
        )
        fig_contour.show()
        fig_contour.write_image("optuna_contour_lr_n_estimators.png", scale=2)
    else:
        print("Skipping contour plot as 'learning_rate' or 'n_estimators' not found in parsed data.")

else:
    print("DataFrame is empty, skipping plotting.")