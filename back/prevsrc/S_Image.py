import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드 및 전처리
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()


# plt.imshow(x_train[0], cmap='gray')
# plt.show()

print(f"x_train shape: {x_train.shape},y_train shape: {y_train.shape}")

def preprocess_data_org(x_train, x_test):
    # 데이터 정규화 및 차원 변환
    x_train = x_train.reshape((60000, 28*28)).astype('float32') / 255.0
    x_test = x_test.reshape((10000, 28*28)).astype('float32') / 255.0
    return x_train, x_test

def preprocess_data(x_train, x_test):
    # CNN을 위해 28x28x1 형태로 차원만 변환합니다.
    # (이미지 개수, 높이, 너비, 채널 수)
    x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
    x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255.0
    return x_train, x_test

x_train, x_test = preprocess_data(x_train, x_test)

# 원-핫 인코딩
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
# ─────────────────────────────────────────
# 고수준 API를 사용한 CNN 모델 구성
# ─────────────────────────────────────────
# 모델 구성
def make_model():
    model = models.Sequential(
        [
            # 1. 컨볼루션 레이어 (특징 추출)
            # 32개의 필터, 3x3 커널, 입력 이미지 형태 지정 (28x28x1)
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            # 2. 풀링 레이어 (크기 축소 및 주요 특징 추출)
            layers.MaxPooling2D((2, 2)),
            # 3. 두 번째 컨볼루션 및 풀링 레이어
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            # 4. 평탄화 레이어 (Dense 레이어를 위해 2D 데이터를 1D로 변환)
            layers.Flatten(),
            # 5. 밀집 레이어 (분류)
            layers.Dense(10, activation='softmax')
        ]
    )
    return model

def execute_hi():
    model = make_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    def train():
        model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    train()


    def evaluate():
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print(f"테스트 세트 정확도: {test_acc:.4f}")
    evaluate()



# ─────────────────────────────────────────
# 저수준 API를 사용한 CNN 모델 구성
# ─────────────────────────────────────────
d# --- [0] 출력 크기 계산 함수 (ksize 포함) ---
def calulate_output_size(input_size, ksize, stride, padding):
    if padding == 'SAME':
        return int(np.ceil(input_size / stride))
    elif padding == 'VALID':
        return int(np.floor((input_size - ksize) / stride)) + 1
    else:
        raise ValueError("Padding must be 'SAME' or 'VALID'")

# --- [1] W3 입력 크기 계산 ---
I0 = 28  # 초기 입력 크기 (28x28)

# Conv1: ksize=3, stride=1, padding='SAME'
osize1 = calulate_output_size(input_size=I0, ksize=3, stride=1, padding='SAME') 

# Pool1: ksize=2, stride=2, padding='SAME'
osize2 = calulate_output_size(input_size=osize1, ksize=2, stride=2, padding='SAME') 

# Conv2: ksize=3, stride=1, padding='SAME'
osize3 = calulate_output_size(input_size=osize2, ksize=3, stride=1, padding='SAME') 

# Pool2: ksize=2, stride=2, padding='SAME'
osize4 = calulate_output_size(input_size=osize3, ksize=2, stride=2, padding='SAME') 

FINAL_SPATIAL_SIZE = osize4      # 7
FINAL_CHANNEL_SIZE = 64          # W2의 출력 채널
W3_INPUT_SIZE = FINAL_SPATIAL_SIZE * FINAL_SPATIAL_SIZE * FINAL_CHANNEL_SIZE

# --- [2] W와 b 정의 (W3에 계산된 변수 적용) ---
W1 = tf.Variable(tf.random.normal([3, 3, 1, 32], stddev=0.01))
W2 = tf.Variable(tf.random.normal([3, 3, 32, 64], stddev=0.01))
# W3에 계산된 W3_INPUT_SIZE 변수 적용
W3 = tf.Variable(tf.random.normal([W3_INPUT_SIZE, 10], stddev=0.01)) 
b1 = tf.Variable(tf.zeros([32]))
b2 = tf.Variable(tf.zeros([64]))
b3 = tf.Variable(tf.zeros([10]))



def model_forward(X):
    #────────────────────────────────────────
    # 첫 번째 컨볼루션 레이어 + ReLU + 맥스 풀링
    conv1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
    relu1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool2d(relu1, ksize=2, strides=2, padding='SAME')
    #────────────────────────────────────────
    # 두 번째 컨볼루션 레이어 + ReLU + 맥스 풀링
    conv2 = tf.nn.conv2d(pool1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2
    relu2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool2d(relu2, ksize=2, strides=2, padding='SAME')
    #────────────────────────────────────────
    # 평탄화
    flat = tf.reshape(pool2, [-1, 7*7*64])

    #────────────────────────────────────────
    # 완전 연결층
    logits = tf.matmul(flat, W3) + b3
    return logits

def cost_fn(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

def train_lo():
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    epochs = 10
    batch_size = 32
    num_batches = x_train.shape[0] // batch_size

    for epoch in range(epochs):
        for i in range(num_batches):
            X_batch = x_train[i*batch_size:(i+1)*batch_size]
            y_batch = y_train[i*batch_size:(i+1)*batch_size]

            with tf.GradientTape() as tape:
                logits = model_forward(X_batch)
                loss = cost_fn(logits, y_batch)

            gradients = tape.gradient(loss, [W1, b1, W2, b2, W3, b3])
            optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2, W3, b3]))

        print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")
def evaluate_lo():
    logits = model_forward(x_test)
    predictions = tf.argmax(logits, axis=1)
    labels = tf.argmax(y_test, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
    print(f"테스트 세트 정확도: {accuracy.numpy():.4f}")

def execute_lo():
    train_lo()
    evaluate_lo()

# execute_lo()

