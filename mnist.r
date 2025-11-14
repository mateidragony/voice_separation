library(reticulate)
reticulate::use_python("/home/mateidragony/.pyenv/versions/3.11.9/bin/python3.11", required = TRUE)

library(tensorflow)
## install_tensorflow(type = "cpu")

library(keras3)
## install_keras()

getModel = function(train, X_train, y_train, epochs) {
    if (train) {
        model = keras_model_sequential() %>%
            layer_dense(units = 256, activation = "relu", input_shape = c(784)) %>%
            layer_dropout(rate = 0.25) %>% 
            layer_dense(units = 128, activation = "relu") %>%
            layer_dropout(rate = 0.25) %>% 
            layer_dense(units = 64, activation = "relu") %>%
            layer_dropout(rate = 0.25) %>%
            layer_dense(units = 10, activation = "softmax")

        model %>% compile(
                      loss = "categorical_crossentropy",
                      optimizer = optimizer_adam(),
                      metrics = c("accuracy")
                  )


        history = model %>% 
            fit(X_train, y_train, epochs = epochs, batch_size = 128, validation_split = 0.15)

        save_model(model, "mnist_model.h5")

        return(model)
    } else {
        model = load_model("mnist_model.h5")
        model %>% compile(
                      loss = "categorical_crossentropy",
                      optimizer = optimizer_adam(),
                      metrics = c("accuracy")
                  )
        return(model)
    }
}

evalModel = function(model, X_test, y_test) {
    model %>% evaluate(X_test, y_test)
}

interactivePredict = function(model, X_test, y_test) {
    for(idx in 1:length(X_test)) {
        img = X_test[idx, , drop = FALSE]
        
        pred = model %>% predict(img)
        digit = which.max(pred) - 1  # because labels are 0–9
        actual = which.max(y_test[idx, , drop = FALSE]) - 1

        cat("Predicted digit:", digit, ", Actual digit:", actual, "\n")

        library(ggplot2)

        image(matrix(mnist$test$x[idx,,], nrow = 28, byrow = TRUE)[, 28:1], 
              col = gray.colors(255), 
              main = paste("Model prediction:", digit, "Actual:", actual))

        if(digit != actual) {
            scan()
        }
    }    
}


flat_norm = function (vec, length, max_val) {
    vec = array_reshape(vec, c(nrow(vec), length))
    vec = vec / max_val
}


num_data = 60000

mnist = dataset_mnist()
X_train = mnist$train$x[1:num_data,,]
X_test = mnist$test$x
y_train = mnist$train$y[1:num_data]
y_test = mnist$test$y

X_train = flat_norm(X_train, 784, 255)
X_test = flat_norm(X_test, 784, 255)

y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)


## model = getModel(FALSE, X_train, y_train, epochs=10)
## evalModel(model, X_test, y_test)
## interactivePredict(model, X_test, y_test)
