require 'rnn'
inputSize=3
LSTMSize=25
outputSize=5
nSteps=3 --sequence length
batchSize=1
lr=0.5
-- test with LSTM
local lstm = nn.LSTM(inputSize, LSTMSize,2*nSteps)
local L=nn.Sequential()
L:add(nn.Linear(LSTMSize,outputSize))
L:add(nn.Sigmoid())
L:training()
local MSE=nn.MSECriterion()
lstm:training()
inputs, outputs, targets, outputsL = {}, {}, {}, {}

--create random data set
for step=1,2*nSteps do
	if(step<=nSteps) then
		inputs[step] = torch.rand(batchSize, inputSize)
		targets[step]=torch.Tensor(batchSize, outputSize):zero()

	else
		inputs[step]=torch.Tensor(batchSize, inputSize):zero()
		targets[step] = torch.rand(batchSize, outputSize)--:mul(2):add(-1)
		targets[step][targets[step]:gt(0.5)]=1
		targets[step][targets[step]:lt(0.5)]=0
	end

end

--train

for iter=1,2000 do
	lstm:forget()
	ld={}
	L:zeroGradParameters()
	lstm:zeroGradParameters()
	--outputsL={}
	--outputs={}
	e=0
	--unfold through time - forward and backward
	for step=1,2*nSteps do

		outputsL[step] = lstm:forward(inputs[step]):clone()
		--clone, otherwise all 'outputs[]' have the reference to the same output - the last one
		outputs[step]= L:forward(outputsL[step]):clone() 
		
		if(step>nSteps) then
		e=e+MSE:forward(outputs[step],targets[step])
		ld[step]=MSE:backward(outputs[step],targets[step])
		end

		local dL
		if step>nSteps then
			-- we want to predict the last nSteps vectors correctly
			dL=L:backward(outputsL[step],ld[step]) 
		else
			-- and we don't care about the predictions that happened before step nSteps - set outputGradients vector to 0
			dL=torch.Tensor(batchSize,LSTMSize):zero() 

		end
		--print(dL)
		dL[dL:gt(5)]=5
		dL[dL:lt(-5)]=-5


		--print('dl',dL)
		lstm:backward(inputs[step], dL)
	end	
	e=e/nSteps

	--if(iter>200) then iter=iter/5; end

	L:updateParameters(lr)	
	lstm:backwardThroughTime()

	--truncate gradient, prevent gradient exploding
	for c=1,#lstm.gradCells do
	lstm.gradCells[c][lstm.gradCells[c]:gt(5)]=5
 	lstm.gradCells[c][lstm.gradCells[c]:lt(-5)]=-5
	--lstm.gradCells[c]:div(lstm.gradCells[c]:norm())
	end
	for c=1,#lstm.gradOutputs do

	lstm.gradOutputs[c][lstm.gradOutputs[c]:gt(5)]=5
 	lstm.gradOutputs[c][lstm.gradOutputs[c]:lt(-5)]=-5
	--lstm.gradOutputs[c]:div(lstm.gradOutputs[c]:norm())
	end
	--for c=1,#lstm.gradInput do
	--lstm.gradInput[c][lstm.gradInput[c]:gt(1)]=1
 	--lstm.gradInput[c][lstm.gradInput[c]:lt(-1)]=-1
	--end
-- 	lstm.gradOutputs[lstm.gradOutputs:gt(1)]=1
-- 	lstm.gradOutputs[lstm.gradOutputs:lt(-1)]=-1
-- 	lstm.gradInput[lstm.gradInput:gt(1)]=1
-- 	lstm.gradInput[lstm.gradInput:lt(-1)]=-1

	--print('-----gradOutput, gradCels[2]-----')
	--print(lstm.gradOutputs[5])
	--print(lstm.gradCells[2])
	print('-----outputs, targets-----')	
	print(outputs[5])
	print(targets[5])
	print('---------')
	print(outputs[6])
	print(targets[6])

	print('Error ',e)
	lstm:updateParameters(lr)

	--collectgarbage()
end
print('Final')
print(outputs[1])
print(targets[1])
print('-----------')
print(outputs[2])
print(targets[2])
print('-----------')
print(outputs[3])
print(targets[3])
print('-----------')
print(outputs[4])
print(targets[4])
print('-----------')
print(outputs[5])
print(targets[5])
print('-----------')
print(outputs[6])
print(targets[6])
