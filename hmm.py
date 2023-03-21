# pip install hmmlearn
import numpy as np
from hmmlearn import hmm



#### Sample data for testing
data = np.random.rand(100, 10)
lengths = [10] * 10
print(len(data[0]))
print(lengths)
#### 

#### Number of hidden states(number of keys to classify)
n_hidden_states = 26
####

# Create a Gaussian HMM
model = hmm.GaussianHMM(n_components=n_hidden_states, covariance_type="diag", n_iter=1000)

# Train the HMM using the sample data
model.fit(data, lengths)

# Predict the hidden states for a new observation sequence
# Replace this with your actual observation sequence
new_observation = np.random.rand(10, 10)
hidden_states = model.predict(new_observation)

print("Hidden states:", hidden_states)
