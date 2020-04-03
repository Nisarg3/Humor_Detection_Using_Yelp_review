model = Sequential()
model.add(Embedding(max_words, 32))
model.add(SimpleRNN(64, dropout=0.1))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc', f1])
history = model.fit(data,labels, epochs=20, batch_size=32, validation_split=0.1)


#plot The Graph
plot_result(history)
