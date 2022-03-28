from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    expectation_true_value = 10
    ug = UnivariateGaussian()
    X = np.random.normal(expectation_true_value, 1, 1000)
    ug.fit(X)
    print("'Question 1:\n'(", ug.mu_, ", ", ug.var_, ")\n", sep='')

    # Question 2 - Empirically showing sample mean is consistent
    sample_size = np.arange(10, 1010, 10)
    estimate_mean = lambda i: np.mean(X[0:i])
    estimations = [estimate_mean(i) for i in sample_size]
    distances = np.absolute(np.array(estimations) - np.full(sample_size.size, expectation_true_value))
    go.Figure(go.Scatter(x=sample_size, y=distances, mode='markers+lines'),
              layout=go.Layout(title="Estimation distance from real value as function of number of samples",
                               xaxis=dict(title='number of samples'),
                               yaxis=dict(title='distance'),
                               height=400, width=800)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_results = ug.pdf(X)
    go.Figure(go.Scatter(x=X, y=pdf_results, mode='markers', marker=dict(size=3)),
              layout=go.Layout(title="pdf of drawn samples values from a N(10,1) distribution",
                               xaxis=dict(title='sample value'),
                               yaxis=dict(title='pdf'),
                               height=500, width=1500)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    orig_mu = np.array([0, 0, 4, 0])
    orig_cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(orig_mu, orig_cov, 1000)
    mg = MultivariateGaussian()
    mg.fit(X)
    print("Question 4:\nestimated expectation:\n", mg.mu_, "\n\nestimated covariance matrix:\n", mg.cov_, sep='')

    # Question 5 - Likelihood evaluation
    ls = np.linspace(-10, 10, 200)
    likelihood = lambda i, j: MultivariateGaussian.log_likelihood(np.array([ls[int(j)], 0, ls[int(i)], 0]), orig_cov, X)
    S = np.fromfunction(np.vectorize(likelihood), (ls.size, ls.size))
    go.Figure(go.Heatmap(x=ls, y=ls, z=S, colorscale='Blues'),
              layout=go.Layout(title="log-likelihood as function of f1 and f3",
                               xaxis=dict(title='f1'),
                               yaxis=dict(title='f3'),
                               height=600, width=600)).show()

    # Question 6 - Maximum likelihood
    max_ind = np.unravel_index(np.argmax(S), np.array(S).shape)
    print('f1 = ', ls[max_ind[1]], '\nf3 = ', ls[max_ind[0]], '\nlog-likelihood = ', S[max_ind])


if __name__ == '__main__':
    np.random.seed(20)
    test_univariate_gaussian()
    test_multivariate_gaussian()
