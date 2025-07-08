| Language | [English](#english)  | [Türkçe](#türkçe) |
|-----------|-------------|---------|

## <a name="english"></a>

# 🚁 Drone Delivery System with Deep Q-Network (DQN)

<div align="center">
  <p><em>Intelligent Package Delivery System with Deep Reinforcement Learning</em></p>
</div>

---

## 📖 About the Project

This project is an intelligent drone simulator that optimizes urban package deliveries using the **Deep Q-Network (DQN)** algorithm. Drones pick up packages from the cargo depot and learn the most efficient routes to deliver them to multiple delivery points through neural network-based decision making.

### 🎯 Project Goals
- Simulate real-world logistics problems with deep learning
- Demonstrate practical application of the DQN algorithm
- Observe the learning process with an interactive visual interface
- Experiment with neural network parameters to achieve the best results

## ✨ Features

- 🧠 **Deep Q-Network (DQN)** with PyTorch neural networks
- 🔄 **Experience Replay** and **Target Network** for stable learning
- 🎮 **Interactive Simulation** with real-time animations  
- 🎯 **Manual & AI Modes** - Control drone or watch AI performance
- 💾 **Model Management** - Save/load trained neural networks (.pth format)
  
### 📊 Simulation Details
- 🟢 **Cargo Depot**: Bottom-right corner where packages are picked up
- 🔴 **Delivery Points**: Randomly placed target locations (1-3 per episode)
- 🔵 **Drone**: Intelligent agent (battery, cargo status display)
- 🔋 **Battery Management**: Different energy costs for movement, takeoff, landing

### 💾 Model Management
- **Save/Load DQN Model**: Store trained neural networks (.pth files)
- **Training Statistics**: Track reward and steps per episode
- **Speed Settings**: Control training and simulation speeds

## 🛠️ Technology Stack

- **Python 3.10+** - Main programming language
- **PyQt5** - GUI framework  
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computations

## 🚀 Installation

```bash
# Python 3.10 or higher is required
python --version

# 1. Clone the Project
git clone https://github.com/FerhatAkalan/DroneDeliverySystemDQN.git
cd DroneDeliverySystemDQN

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Start the Simulator
python drone_delivery_system.py
```

## 🧠 Deep Q-Network Algorithm

### 🏗️ Neural Network Architecture
- **Input Layer**: 12 neurons (state vector)
- **Hidden Layer**: 128 neurons + ReLU + Dropout(0.2)
- **Output Layer**: 6 neurons (Q-values for actions)

### 📊 State Representation (12D Vector)
The DQN uses a 12-dimensional state vector containing:
- **Drone Position**: Normalized x, y coordinates (0-1)
- **Drone Status**: Has cargo (0/1), is flying (0/1)
- **Battery Level**: Normalized battery percentage (0-1)
- **Cargo Depot**: Normalized x, y coordinates (0-1)
- **Target Information**: Nearest delivery point position and distance
- **Environmental Data**: Target count, step count, done flags

### 🔄 Action Space
| Action | Description | Condition |
|--------|-------------|-----------|
| `0` ⬇️ | Move down | Drone must be flying |
| `1` ➡️ | Move right | Drone must be flying |
| `2` ⬆️ | Move up | Drone must be flying |
| `3` ⬅️ | Move left | Drone must be flying |
| `4` 📦 | Pick/Drop cargo | Drone must be on the ground |
| `5` 🛫🛬 | Takeoff/Land | Always available |

### 🏆 Optimized Reward System

#### ✅ Positive Rewards
- **Pick up cargo**: +200 points (increased for better learning)
- **Successful delivery**: +1000 points (significantly increased)
- **Task completion**: +500 points + battery bonus
- **Approaching target**: +10 points per step (doubled)
- **Near depot bonus**: +15 points when close to depot and grounded
- **Correct action bonus**: +10 points for right action at right place

#### ❌ Penalties (Reduced for Better Learning)
- **Invalid action**: -0.5 to -2 points (much reduced)
- **Move away from target**: -1 point per step (halved)
- **Battery depletion**: -50 points
- **Timeout**: -20 points
- **Movement cost**: -0.2 points per move (minimal)

## 📸 Project Image
![Ekran görüntüsü 2025-07-08 115233](https://github.com/user-attachments/assets/e5330c90-afcf-4b35-9ab4-657bd4782525)


### ⚙️ DQN Hyperparameters
- **Learning Rate**: 0.001
- **Discount Factor (γ)**: 0.99
- **Epsilon Decay**: 0.995
- **Batch Size**: 32
- **Replay Buffer Size**: 10,000
- **Target Network Update**: Every 100 steps
- **Default Training Episodes**: 20,000

## 🔬 Experimental Results

### 📊 Training Performance
- **Grid Size**: 5x5 (default optimal setting)
- **Recommended Training**: 20,000 episodes for stable performance
- **Success Rate**: 85%+ after proper training
- **State Representation**: 12D vector for efficient learning

### 📈 Learning Curve
As training progresses, drone performance increases significantly:
- **First 5,000 episodes**: Random exploration behavior
- **5,000-15,000 episodes**: Learning pickup and delivery strategy
- **15,000+ episodes**: Optimized route planning and battery management

### 🎯 Performance Metrics
- **Average Reward**: Steadily increases from negative to positive values
- **Episode Length**: Decreases as drone learns efficient paths
- **Battery Usage**: Optimized energy consumption patterns
- **Success Rate**: Reaches 90%+ completion rate after full training

## 🤝 Contributing

If you want to contribute to this project:

1. **Fork** it
2. **Create a feature branch** (`git checkout -b feature/NewFeature`)
3. **Commit** (`git commit -am 'Add new feature'`)
4. **Push** (`git push origin feature/NewFeature`)
5. **Create a Pull Request**

### 💡 Contribution Areas

| 🚀 Feature | 📝 Description | 🎯 Difficulty |
|-----------|---------------|--------------|
| **Obstacle System** | Add obstacles and environmental hazards | 🟡 Medium |
| **Multi-Agent DQN** | Multiple drones learning simultaneously | 🔴 Hard |
| **Double DQN** | Improved DQN with double Q-learning | � Medium |
| **3D Visualization** | 3D environment rendering | 🟡 Medium |
| **Real-Time Statistics** | Live training metrics with Matplotlib | 🟢 Easy |
| **Advanced Rewards** | Complex reward shaping and curriculum learning | � Hard |

---

<div align="center">
  <h3>⭐ If you like the project, don't forget to give a star! ⭐</h3>  
  <p>If you have any questions, you can <a href="https://github.com/FerhatAkalan/DroneDeliverySystemDQN/issues">open an issue</a>.</p>
  
  **🚁 Happy Coding! 🚁**
  
  <sub>Made with ❤️ by Ferhat Akalan</sub>
</div>

---

</br>

---

## <a name="türkçe"></a>

# 🚁 Deep Q-Network ile Drone Teslimat Sistemi

<div align="center">
  <p><em>Derin Pekiştirmeli Öğrenme ile Akıllı Paket Teslimat Sistemi</em></p>
</div>

---

## 📖 Proje Hakkında

Bu proje, **Deep Q-Network (DQN)** algoritması kullanarak şehir içi paket teslimatlarını optimize eden akıllı bir dron simülatörüdür. Dronlar, neural network tabanlı karar verme ile kargo deposundan paketleri alıp birden fazla teslimat noktasına en verimli rotaları öğrenerek ulaştırırlar.

### 🎯 Proje Hedefleri
- Gerçek dünya lojistik problemlerini derin öğrenme ile simüle etmek
- DQN algoritmasının pratik uygulamasını göstermek  
- İnteraktif görsel arayüz ile öğrenme sürecini gözlemlemek
- Neural network parametrelerini deneyimleyerek en iyi sonuçları elde etmek          

## ✨ Özellikler

- 🧠 **Deep Q-Network (DQN)** PyTorch neural network'leri ile
- 🔄 **Experience Replay** ve **Target Network** ile kararlı öğrenme
- 🎮 **İnteraktif Simülasyon** gerçek zamanlı animasyonlarla
- 🎯 **Manuel & AI Modları** - Drone kontrolü veya AI performansı izleme
- 💾 **Model Yönetimi** - Eğitilmiş neural network'leri kaydetme/yükleme (.pth format)

### 📊 Simülasyon Detayları
- 🟢 **Kargo Deposu**: Sağ alt köşede paketlerin alındığı merkez
- 🔴 **Teslimat Noktaları**: Rastgele yerleştirilen hedef lokasyonlar (episode başına 1-3 adet)
- 🔵 **Drone**: Akıllı ajan (batarya, kargo durumu gösterimi)
- 🔋 **Batarya Yönetimi**: Hareket, kalkış, iniş için farklı enerji maliyetleri

## 🛠️ Teknoloji Stack

- **Python 3.10+** - Ana programlama dili
- **PyQt5** - GUI framework
- **PyTorch** - Derin öğrenme framework'ü
- **NumPy** - Sayısal hesaplamalar

## 🚀 Kurulum

```bash
# Python 3.10 veya üzeri gereklidir
python --version

# 1. Projeyi Klonlayın
git clone https://github.com/FerhatAkalan/DroneDeliverySystemDQN.git
cd DroneDeliverySystemDQN

# 2. Bağımlılıkları Yükleyin
pip install -r requirements.txt

# 3. Simülatörü Başlatın
python drone_delivery_system.py
```

## 🧠 Deep Q-Network Algoritması

### 🏗️ Neural Network Mimarisi
- **Giriş Katmanı**: 12 nöron (durum vektörü)
- **Gizli Katman**: 128 nöron + ReLU + Dropout(0.2)
- **Çıkış Katmanı**: 6 nöron (Q-değerleri)

### 📊 Durum Temsili (12 Boyutlu Vektör)
DQN şu bilgileri içeren 12 boyutlu durum vektörü kullanır:
- **Drone Pozisyonu**: Normalize edilmiş x, y koordinatları (0-1)
- **Drone Durumu**: Kargo var mı (0/1), uçuyor mu (0/1)
- **Batarya Seviyesi**: Normalize edilmiş batarya yüzdesi (0-1)
- **Kargo Deposu**: Normalize edilmiş x, y koordinatları (0-1)
- **Hedef Bilgisi**: En yakın teslimat noktası pozisyonu ve mesafe
- **Çevresel Veri**: Hedef sayısı, adım sayısı, tamamlanma durumu

### 🔄 Eylem Uzayı
| Eylem | Açıklama | Koşul |
|-------|----------|-------|
| `0` ⬇️ | Aşağı hareket | Drone havada olmalı |
| `1` ➡️ | Sağa hareket | Drone havada olmalı |
| `2` ⬆️ | Yukarı hareket | Drone havada olmalı |
| `3` ⬅️ | Sola hareket | Drone havada olmalı |
| `4` 📦 | Kargo al/bırak | Drone yerde olmalı |
| `5` 🛫🛬 | Kalkış/İniş | Her zaman kullanılabilir |

### 🏆 Ödül Sistemi

#### ✅ Pozitif Ödüller
- **Kargo alma**: +100 puan
- **Başarılı teslimat**: +500 puan
- **Görev tamamlama**: +1000 puan + batarya bonusu
- **Hedefe yaklaşma**: +5 puan/adım
- **Hedefe ulaşma**: Mesafe tabanlı bonus

#### ❌ Cezalar
- **Geçersiz eylem**: -1 ile -10 puan arası
- **Hedeften uzaklaşma**: -2 puan/adım
- **Batarya bitimi**: -50 puan
- **Zaman aşımı**: -20 puan
- **Sınır ihlali**: -1 puan
  
## 📸 Proje Görseli
![Ekran görüntüsü 2025-07-08 115233](https://github.com/user-attachments/assets/e5330c90-afcf-4b35-9ab4-657bd4782525)

### ⚙️ DQN Hiperparametreleri
- **Öğrenme Oranı**: 0.001
- **İndirgeme Faktörü (γ)**: 0.99
- **Epsilon Azalma**: 0.995
- **Batch Boyutu**: 32
- **Replay Buffer Boyutu**: 10,000
- **Target Network Güncelleme**: Her 100 adımda
- **Varsayılan Eğitim Episode**: 20,000

## 🔬 Deneysel Sonuçlar

### 📊 Eğitim Performansı
- **Grid Boyutu**: 5x5 (varsayılan optimal ayar)
- **Önerilen Eğitim**: Kararlı performans için 20,000 episode
- **Başarı Oranı**: Uygun eğitim sonrası %85+
- **Durum Temsili**: Verimli öğrenme için 12 boyutlu vektör

### 📈 Öğrenme Eğrisi
Eğitim ilerledikçe dronun performansı belirgin şekilde artar:
- **İlk 5,000 episode**: Rastgele keşif davranışı
- **5,000-15,000 episode**: Kargo alma ve teslimat stratejisi öğrenme
- **15,000+ episode**: Optimize edilmiş rota planlama ve batarya yönetimi

### 🎯 Performans Metrikleri
- **Ortalama Ödül**: Negatif değerlerden pozitif değerlere sürekli artış
- **Episode Süresi**: Drone verimli yollar öğrendikçe azalır
- **Batarya Kullanımı**: Optimize edilmiş enerji tüketim kalıpları
- **Başarı Oranı**: Tam eğitim sonrası %90+ tamamlanma oranı

## 🤝 Katkıda Bulunma

Bu projeye katkıda bulunmak istiyorsanız:

1. **Fork** edin
2. **Feature branch** oluşturun (`git checkout -b feature/NewFeature`)
3. **Commit** edin (`git commit -am 'Add new feature'`)
4. **Push** edin (`git push origin feature/NewFeature`)
5. **Pull Request** oluşturun

### 💡 Katkı Alanları

| 🚀 Özellik | 📝 Açıklama | 🎯 Zorluk |    
|-----------|-------------|----------|    
| **Engel Sistemi** | Engeller ve çevresel tehlikeler ekleme | 🟡 Orta |    
| **Çok Ajanlı DQN** | Aynı anda birden fazla drone öğrenmesi | 🔴 Zor |    
| **Double DQN** | Çifte Q-öğrenme ile geliştirilmiş DQN | � Orta |    
| **3D Görselleştirme** | 3D ortam render etme | 🟡 Orta |    
| **Gerçek Zamanlı İstatistikler** | Matplotlib ile canlı eğitim metrikleri | 🟢 Kolay |    
| **Gelişmiş Ödüller** | Karmaşık ödül şekillendirme ve müfredat öğrenmesi | � Zor |

---

<div align="center">
  <h3>⭐ Projeyi beğendiyseniz star vermeyi unutmayın! ⭐</h3>
  <p>Herhangi bir sorunuz varsa <a href="https://github.com/FerhatAkalan/DroneDeliverySystemDQN/issues">issue açabilirsiniz</a>.</p>
  
  **🚁 Happy Coding! 🚁**
  
  <sub>Made with ❤️ by Ferhat Akalan</sub>
</div>
