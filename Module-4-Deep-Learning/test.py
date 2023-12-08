input = minitorch.rand((16,))

output = avgpool(avgpool(input, 2), 4) # Size of average pool is second arg. 

output.shape 

output[1].backward()