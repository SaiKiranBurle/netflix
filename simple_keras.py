from keras import Model
from keras.backend import clip
from keras.layers import Input, Embedding, concatenate, Dense, Flatten, Lambda

from data_generator import NUM_USERS, NUM_MOVIES, get_ratings_data, BATCH_SIZE, transform_ratings_into_classes

classes = False


def get_class_model():
    user = Input(shape=(1,), dtype='int32', name='input_user')
    user_embedding = Embedding(output_dim=25, input_dim=NUM_USERS)(user)

    movie = Input(shape=(1,), dtype='int32', name='input_movie')
    movie_embedding = Embedding(output_dim=25, input_dim=NUM_MOVIES)(movie)

    merged = concatenate([user_embedding, movie_embedding])

    x = Flatten()(merged)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    ratings = Dense(5, activation='sigmoid', name='ratings')(x)

    model = Model(inputs=[user, movie], outputs=[ratings])
    # TODO: Maybe some other metrics?
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_regression_model():
    user = Input(shape=(1,), dtype='int32', name='input_user')
    user_embedding = Embedding(output_dim=25, input_dim=NUM_USERS)(user)

    movie = Input(shape=(1,), dtype='int32', name='input_movie')
    movie_embedding = Embedding(output_dim=25, input_dim=NUM_MOVIES)(movie)

    merged = concatenate([user_embedding, movie_embedding])

    x = Flatten()(merged)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    ratings = Dense(1)(x)
    ratings = Lambda(lambda y: clip(y, 0., 5.), name='ratings')(ratings)

    model = Model(inputs=[user, movie], outputs=[ratings])
    # TODO: Maybe some other metrics?
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

    return model

if __name__ == "__main__":
    if classes:
        model = get_class_model()
        train_users, train_movies, train_ratings = get_ratings_data()
        train_ratings = transform_ratings_into_classes(train_ratings)
    else:
        model = get_regression_model()
        train_users, train_movies, train_ratings = get_ratings_data()
    model.fit({'input_user': train_users, 'input_movie': train_movies},
              {'ratings': train_ratings},
              epochs=5, batch_size=BATCH_SIZE,
              validation_split=0.2)
