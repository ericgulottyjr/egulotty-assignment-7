from flask import Flask, render_template, request, url_for, session
import secrets
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import t, norm

app = Flask(__name__)
app.secret_key = 'test-key'


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # TODO 1: Generate a random dataset X of size N with values between 0 and 1
    X = np.random.rand(N)

    # TODO 2: Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    error = np.random.normal(loc=mu, scale=np.sqrt(sigma2), size=N)
    Y = beta0 + beta1 * X + error

    # TODO 3: Fit a linear regression model to X and Y
    model = LinearRegression().fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # TODO 4: Generate a scatter plot of (X, Y) with the fitted regression line
    plot1_path = "static/plot1.png"

    plt.scatter(X, Y, color="blue")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot with Regression Line")
    plt.savefig(plot1_path)
    plt.close()

    # TODO 5: Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        X_sim = np.random.rand(N)
        Y_sim = beta0 + beta1 * X_sim + np.random.normal(loc=mu, scale=np.sqrt(sigma2), size=N)
        sim_model = LinearRegression().fit(X_sim.reshape(-1, 1), Y_sim)
        sim_slope = sim_model.coef_[0]
        sim_intercept = sim_model.intercept_

        slopes.append(sim_slope)
        
        intercepts.append(sim_intercept)


    # TODO 8: Plot histograms of slopes and intercepts
    plot2_path = "static/plot2.png"

    plt.hist(slopes, bins=20, alpha=0.7, label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.7, label="Intercepts")
    plt.axvline(slope, color='blue', linestyle='--', label=f'Slope: {slope:.2f}')
    plt.axvline(intercept, color='orange', linestyle='--', label=f'Intercept: {intercept:.2f}')
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(plot2_path)
    plt.close()

    # TODO 9: Return data needed for further analysis, including slopes and intercepts
    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = np.mean(np.abs(slopes) > abs(slope))
    intercept_extreme = np.mean(np.abs(intercepts) > abs(intercept))


    # Return data needed for further analysis
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    # Get user input from the form
    N = int(request.form["N"])
    mu = float(request.form["mu"])
    sigma2 = float(request.form["sigma2"])
    beta0 = float(request.form["beta0"])
    beta1 = float(request.form["beta1"])
    S = int(request.form["S"])

    # Generate data and initial plots
    (
        X,
        Y,
        slope,
        intercept,
        plot1,
        plot2,
        slope_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    ) = generate_data(N, mu, beta0, beta1, sigma2, S)

    # Store data in session
    session["X"] = X.tolist()
    session["Y"] = Y.tolist()
    session["slope"] = slope
    session["intercept"] = intercept
    session["slopes"] = slopes
    session["intercepts"] = intercepts
    session["slope_extreme"] = slope_extreme
    session["intercept_extreme"] = intercept_extreme
    session["N"] = N
    session["mu"] = mu
    session["sigma2"] = sigma2
    session["beta0"] = beta0
    session["beta1"] = beta1
    session["S"] = S

    # Return render_template with variables
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        slope_extreme=slope_extreme,
        intercept_extreme=intercept_extreme,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
        parameter_label = "Slope"
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0
        parameter_label = "Intercept"

    # Calculate p-value based on test type
    if test_type == "!=":
        # Two-sided test
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= np.abs(observed_stat - hypothesized_value))
    elif test_type == ">":
        # Greater than test
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "<":
        # Less than test
        p_value = np.mean(simulated_stats <= observed_stat)
    else:
        p_value = None  # Handle invalid test type

    # Fun message for small p-values
    fun_message = "Whoa! This result is highly significant!" if p_value and p_value <= 0.0001 else ""

    # Plot histogram of simulated statistics
    plot3_path = "static/plot3.png"

    plt.hist(simulated_stats, bins=20, alpha=0.7)
    plt.axvline(observed_stat, color="red", linestyle="--", label=f"Observed {parameter_label}: {observed_stat:.3f}")
    plt.axvline(hypothesized_value, color="green", linestyle="--", label=f"Hypothesized {parameter_label}: {hypothesized_value:.2f}")
    plt.xlabel(f"{parameter_label} Values")
    plt.title(f"Histogram of Simulated {parameter_label}s")
    plt.legend()
    plt.savefig(plot3_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve parameters from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    observed_slope = float(session.get("slope"))
    observed_intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    # Use the stored simulations
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = observed_slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = observed_intercept
        true_param = beta0

    # Confidence Interval Calculations
    mean_estimate = np.mean(estimates)
    std_error = np.std(estimates, ddof=1) / np.sqrt(len(estimates))
    alpha = 1 - (confidence_level / 100)
    degrees_of_freedom = len(estimates) - 1
    t_crit = t.ppf(1 - alpha / 2, df=degrees_of_freedom)
    margin_of_error = std_error * t_crit
    ci_lower = mean_estimate - margin_of_error
    ci_upper = mean_estimate + margin_of_error
    includes_true = ci_lower <= true_param <= ci_upper

    # Plot the confidence interval graph as per your requirements
    plot4_path = "static/plot4.png"

    plt.figure(figsize=(10, 5))  # Adjust height since y-axis is less important

    # Plot simulated estimates as gray points along y=0
    y_values = np.zeros_like(estimates)
    plt.scatter(estimates, y_values, color="gray", alpha=0.5, label="Simulated Estimates")

    # Plot the mean estimate as a blue dot at y=0
    mean_color = "green" if includes_true else "red"
    plt.scatter(mean_estimate, 0, color=mean_color, label="Mean Estimate", s=100)

    # Plot the confidence interval as a horizontal line
    plt.hlines(y=0, xmin=ci_lower, xmax=ci_upper, colors="blue", linestyles="-", linewidth=5, label=f"{confidence_level}% Confidence Interval")

    # Plot the true parameter as a vertical green dashed line
    plt.axvline(true_param, color="green", linestyle="--", label="True Parameter")

    # Hide y-axis ticks as they are not needed
    plt.yticks([])

    # Add labels and title
    plt.xlabel(f"Estimated {parameter.capitalize()}")
    plt.title(f"{confidence_level}% Confidence Interval for {parameter.capitalize()}")

    # Add legend
    plt.legend(loc='upper left')

    plt.savefig(plot4_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )

if __name__ == "__main__":
    app.run(debug=True)