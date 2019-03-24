import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
import os
import random
import matplotlib.pyplot as plt

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(176, 200, 3),
                 activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

opt = keras.optimizers.adam(lr=0.0001, decay=1e-6)
opt2 = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt2,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/attack")

train_data_dir = "train_data"

acc_values = []
loss_values = []

def check_data():
    choices = {"No attack": no_attacks,
               "Attack main base": attack_main_base,
               "Attack Enemy structure": attack_enemy_structures,
               "Attack Eneemy units": attack_enemy_units}

    total_data = 0

    lengths = []
    for choice in choices:
        #print("Length of {} is: {}".format(choice, len(choices[choice])))
        total_data += len(choices[choice])
        lengths.append(len(choices[choice]))

    #print("Total data length now is:", total_data)
    return lengths

hm_epochs = 50

for i in range(hm_epochs):
    print("EPOCHS : ",i)
    all_files = os.listdir(train_data_dir)
    random.shuffle(all_files)

    no_attacks = []
    attack_main_base = []
    attack_enemy_structures = []
    attack_enemy_units = []

    for file in all_files:
        full_path = os.path.join(train_data_dir, file)
        data = np.load(full_path)
        data = list(data)
        for d in data:
            choice = np.argmax(d[0])
            if choice == 0:
                no_attacks.append([d[0], d[1]])
            elif choice == 1:
                attack_main_base.append([d[0], d[1]])
            elif choice == 2:
                attack_enemy_structures.append([d[0], d[1]])
            elif choice == 3:
                attack_enemy_units.append([d[0], d[1]])

    lengths = check_data()
    lowest_data = min(lengths)

    random.shuffle(no_attacks)
    random.shuffle(attack_main_base)
    random.shuffle(attack_enemy_structures)
    random.shuffle(attack_enemy_units)

    no_attacks = no_attacks[:lowest_data]
    attack_main_base = attack_main_base[:lowest_data]
    attack_enemy_structures = attack_enemy_structures[:lowest_data]
    attack_enemy_units = attack_enemy_units[:lowest_data]

    check_data()

    train_data = no_attacks + attack_main_base + attack_enemy_structures + attack_enemy_units

    random.shuffle(train_data)
    print(len(train_data))

    test_size = 1
    batch_size = 128

    x_train = np.array([i[1] for i in train_data[:-test_size]]).reshape(-1, 176, 200, 3)
    y_train = np.array([i[0] for i in train_data[:-test_size]])

    history = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 shuffle=True,
                 verbose=1, callbacks=[tensorboard])
    history_dict = history.history
    tmp_acc = history_dict['acc']
    tmp_loss = history_dict['loss']
    acc_values.append(tmp_acc[0])
    loss_values.append(tmp_loss[0])
    model.save("Model-{}-epochs-{}-attack".format(hm_epochs, 0.0001))

#Display result
acc_values = np.asarray(acc_values)
acc_values.shape = (np.size(acc_values), 1)
hm_epochs = range(1, len(acc_values) + 1)
plt.plot(hm_epochs, acc_values, 'b', label='Training acc')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

loss_values = np.asarray(loss_values)
loss_values.shape = (np.size(loss_values), 1)
hm_epochs = range(1, len(loss_values) + 1)
plt.plot(hm_epochs, loss_values, 'b', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
