import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

f = open('./data/jena_climate_2009_2016.csv')
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
print("lines[0] "+'*'*20)
print(lines[0],len(lines))

float_data = np.zeros((len(lines),len(header)-1))
for i,line in enumerate(lines):
    float_data[i] = [float(x) for x in line.split(",")[1:]]

print("float_data[0] "+'*'*20)
print(float_data[0],float_data[0][0])

# temp = float_data[:,1]
# plt.plot(range(len(temp)),temp)
# plt.show()

mean = float_data[:200000].mean(axis=0)
float_data -= mean
print("mean "+'*'*20)
print(mean)
std = float_data[:200000].std(axis=0)
float_data /= std
print("std "+'*'*20)
print(std)

print("float_data.shape "+"*"*20)
print(float_data.shape)

def generator(data,lookback,delay,min_index,max_index,shuffle=False,batch_size=128,step=6):
    if max_index is None:
        max_index = len(data) - delay -1
    i = min_index+lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index+lookback, max_index, size=batch_size)
        else:
            if i+batch_size>=max_index:
                i = min_index+lookback
            rows = np.arange(i,min(i+batch_size,max_index))
            i += len(rows)
        
        samples = np.zeros((len(rows), lookback//step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j,row in enumerate(rows):
            indices = range(rows[j] - lookback,rows[j],step)
            samples[j] = data[indices]
            targets[j] = data[rows[j]+delay][1]
        yield samples,targets

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=200000, shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200000, max_index=300000, step=step, batch_size=batch_size)
test_gen = generator(float_data,lookback=lookback, delay=delay, min_index=300000, max_index=None, step=step, batch_size=batch_size)
val_steps = (300000-200001-lookback)//batch_size
test_steps = (len(float_data)-300001-lookback)//batch_size

model = Sequential()
model.add(layers.GRU(32, 
                     dropout=0.2, 
                     recurrent_dropout=0.2, 
                     return_sequences=True,
                     input_shape=(None,float_data.shape[-1])))
model.add(layers.GRU(64,
                     activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, 
                              steps_per_epoch=500, 
                              epochs=30, 
                              validation_data=val_gen, 
                              validation_steps=val_steps)

his = history.history
loss = his['loss']
val_loss = his['val_loss']
axis = range(1, len(loss)+1)
plt.plot(axis, loss, 'bo', label="Training loss")
plt.plot(axis, val_loss, 'b', label="Validation loss")
plt.legend()
plt.show()
