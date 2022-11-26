'''
model-free vs model-based

model-free - the model does not try to have an understanding of its surroundings. It
receives the observations and computes an action directly based on the observation.

model-based - the method tries to 'understand' its surroundings by trying to predict the
next observation or reward will be. Based on this prediction the model tries to find
the best possible action and making predictions multiple times and looking more and
more steps into the future.

Conclusion:
Both classes have strong and weak sides. In deterministic environments with strict
rules purely madel based models are used. In complex environments with many rich observations
it is often not possible to use model-based and rather model-free models are used
to directly compute the actions based on the observations.
Recently researchers started to mix the benefits from both worlds. See
DeepMind research paper on imagination in agents.

value based vs policy based


on-policy vs off-policy
'''