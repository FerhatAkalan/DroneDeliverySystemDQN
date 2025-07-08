# -*- coding: utf-8 -*-
"""
===============================================================================
🚁 DRONE PAKET TESLİMAT SİSTEMİ - DEEP Q-NETWORK (DQN) İLE PEKİŞTİRMELİ ÖĞRENME
===============================================================================
Final Projesi - Ferhat Akalan

📋 PROJE AÇIKLAMASI:
Bu proje, drone'ların şehir içi paket teslimatlarında Deep Q-Network (DQN) algoritması 
kullanarak optimal strateji öğrenmesini simüle eder.

🎯 SENARYO:
- 100 şarjlı drone, 5x5 grid tabanlı ortamda hareket eder
- Bir kargo deposu (🟢 yeşil kare) sağ alt köşede bulunur
- Her episode'da 1-3 arası rastgele teslimat noktası (🔴 kırmızı daireler) belirir
- Drone görevi: kargo deposuna gidip kargo almalı → en yakın teslimat noktasına teslim etmeli

🧠 DQN ALGORİTMASI ÖZELLİKLERİ:
- Experience Replay Buffer: Geçmiş deneyimlerden batch öğrenme (buffer_size=10,000)
- Target Network: Kararlı eğitim için ayrı hedef ağı (update_freq=100 step)
- Epsilon-Greedy Exploration: Keşif dengesi (1.0 → 0.01)
- Optimizasyon edilmiş reward sistemi: Pozitif pekiştirme odaklı
- 12 boyutlu state representation: Verimli durum temsili

🏗️ NEURAL NETWORK MİMARİSİ:
- Input Layer: 12 nöron (state vector)
- Hidden Layer 1: 128 nöron + ReLU + Dropout(0.2)
- Output Layer: 6 nöron (Q-values for actions)

⚙️ TEKNİK DETAYLAR:
- PyTorch ile DQN implementasyonu
- PyQt5 ile interaktif görsel arayüz
- Thread-based eğitim (UI donmaması için)
- Real-time neural network görselleştirmesi
- Model kaydet/yükle fonksiyonları (.pth format)

Bu proje, reinforcement learning'in gerçek dünya uygulamalarına bir örnektir.
Drone'lar bu algoritmalarla otonom olarak karar verebilir hale gelir.
"""

# =====================
# KÜTÜPHANE İMPORTLARI
# =====================
import sys
import os
import random
import pickle
import numpy as np

# PyTorch Deep Learning kütüphaneleri
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# PyQt5 GUI kütüphaneleri
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, 
                             QDoubleSpinBox, QGroupBox, QGridLayout, QFileDialog, 
                             QMessageBox, QComboBox, QSlider)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QFont, QIcon, QPixmap

# =====================
# ORTAM (ENVIRONMENT) SINIFI
# =====================
class DroneDeliveryEnv:
    """
    ========================================================================
    DRONE TESLİMAT ORTAMI (REINFORCEMENT LEARNING ENVIRONMENT)
    ========================================================================
    OpenAI Gym benzeri ortam yapısı:
    - Grid tabanlı 2D dünya (5x5, 6x6, 7x7 boyutlarında)
    - State: 12 boyutlu vektör (drone pozisyonu, kargo durumu, hedef bilgisi)
    - Action: 6 farklı eylem (hareket, kargo alma/bırakma, kalkış/iniş)
    - Reward: Optimizasyon edilmiş ödül sistemi
    
    Görsel Kodlama:
    - 🟢 Yeşil: Kargo deposu
    - 🔴 Kırmızı: Teslimat noktaları
    - 🔵 Mavi: Drone
    - 🔋 Batarya seviyesi
    """
    def __init__(self, grid_size=5, max_steps=100, n_deliveries=1):
        """
        Ortam parametrelerini başlat
        
        Args:
            grid_size (int): Grid boyutu (3-7 arası)
            max_steps (int): Maksimum adım sayısı
            n_deliveries (int): Teslimat noktası sayısı (1-3 arası)
        """
        # === TEMEL PARAMETRELER ===
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_deliveries = n_deliveries
        
        # === EYLEM UZAYI ===
        # 6 farklı eylem: hareket (4) + kargo işlemi (1) + uçuş kontrolü (1)
        self.action_space_n = 6
        # Eylem açıklamaları:
        # 0: Aşağı ⬇️, 1: Sağa ➡️, 2: Yukarı ⬆️, 3: Sola ⬅️
        # 4: Kargo Al/Bırak 📦, 5: Kalk/İn 🛫🛬
        
        # === SABİT TESLİMAT NOKTALARı ===
        # Grid'in 4 köşesi (Taxi-v3 problemi benzeri)
        self.fixed_delivery_points = [
            np.array([0, 0]),                    # Sol üst köşe
            np.array([0, self.grid_size-1]),     # Sağ üst köşe
            np.array([self.grid_size-1, 0]),     # Sol alt köşe
            np.array([self.grid_size-1, self.grid_size-1])  # Sağ alt köşe
        ]
        
        # === BATARYA TÜKETİM ORANLARI ===
        self.move_battery_cost = 1      # Normal hareket: 1 birim
        self.takeoff_battery_cost = 3   # Kalkış: 3 birim (azaltıldı)
        self.landing_battery_cost = 3   # İniş: 3 birim (azaltıldı)
        
        # Ortamı ilk duruma getir
        self.reset()

    def reset(self):
        """
        ===============================================
        ORTAMI SIFIRLA (YENİ EPİSODE BAŞLAT)
        ===============================================
        Her episode başında ortamı başlangıç durumuna getirir.
        Rastgele drone pozisyonu ve teslimat noktaları oluşturur.
        
        Returns:
            np.ndarray: Başlangıç state vektörü (12 boyutlu)
        """
        # === DRONE BAŞLANGIÇ POZİSYONU ===
        # Drone'u grid üzerinde rastgele konumlandır
        self.drone_pos = np.array([
            random.randint(0, self.grid_size-1),
            random.randint(0, self.grid_size-1)
        ])
        
        # === KARGO DEPOSU POZİSYONU ===
        # Kargo deposu her zaman sağ alt köşede (sabit)
        self.cargo_depot_pos = np.array([self.grid_size-1, self.grid_size-1])
        
        # === TESLİMAT NOKTALARINI SEÇ ===
        # Her episode'da 1-3 arası rastgele teslimat noktası
        self.n_deliveries = random.randint(1, 3)
        
        # Kargo deposu dışındaki köşelerden teslimat noktalarını seç
        available_indices = [i for i in range(len(self.fixed_delivery_points)) 
                           if not np.array_equal(self.fixed_delivery_points[i], self.cargo_depot_pos)]
        chosen_indices = random.sample(available_indices, self.n_deliveries)
        self.delivery_points = [self.fixed_delivery_points[i].copy() for i in chosen_indices]
        self.delivery_indices = chosen_indices
        
        # === DRONE DURUMU ===
        self.has_cargo = False          # Kargo taşıyor mu?
        self.battery = 100              # Batarya seviyesi (100% full)
        self.steps = 0                  # Adım sayacı
        self.delivered = [False] * len(self.delivery_points)  # Teslimat durumları
        self.done = False               # Episode bitmiş mi?
        self.is_flying = False          # Drone uçuyor mu?
        
        # === GÖRSEL ANİMASYON DEĞİŞKENLERİ ===
        self.landing_state = "landed"           # İniş/kalkış durumu
        self.landing_animation_step = 0         # Animasyon adımı
        
        # === ÖDÜL TAKİP DEĞİŞKENLERİ ===
        self.last_reward = 0            # Son adımdaki ödül
        self.total_reward = 0           # Episode toplam ödülü
        
        return self.get_state()         # Başlangıç state'ini döndür

    def get_state(self):
        """
        ===============================================
        STATE REPRESENTATION (DURUM TEMSİLİ)
        ===============================================
        DQN için optimize edilmiş 12 boyutlu state vektörü oluşturur.
        Bu vektör, neural network'ün karar vermesi için gerekli tüm bilgiyi içerir.
        
        State Vektörü İçeriği:
        [0-1]: Drone pozisyonu (x, y) - normalize edilmiş
        [2-3]: Drone durumları (kargo_var_mı, uçuyor_mu) - binary
        [4]:   Batarya seviyesi - normalize edilmiş (0-1)
        [5-6]: Kargo deposu pozisyonu - normalize edilmiş
        [7-8]: Hedef pozisyonu - normalize edilmiş
        [9]:   Hedefe mesafe - normalize edilmiş
        [10]:  Teslimat tamamlanma oranı
        [11]:  Kalan adım oranı
        
        Returns:
            np.ndarray: 12 boyutlu float32 state vektörü
        """
        state = np.zeros(12, dtype=np.float32)
        
        # === DRONE POZİSYONU (Normalize edilmiş) ===
        state[0] = self.drone_pos[0] / (self.grid_size - 1) if self.grid_size > 1 else 0
        state[1] = self.drone_pos[1] / (self.grid_size - 1) if self.grid_size > 1 else 0
        
        # === DRONE DURUMLARI (Binary) ===
        state[2] = float(self.has_cargo)    # 1.0 if kargo var, 0.0 if yok
        state[3] = float(self.is_flying)    # 1.0 if uçuyor, 0.0 if yerde
        
        # === BATARYA SEVİYESİ (Normalize edilmiş) ===
        state[4] = self.battery / 100.0     # 0.0 - 1.0 arası
        
        # === KARGO DEPOSU POZİSYONU (Normalize edilmiş) ===
        state[5] = self.cargo_depot_pos[0] / (self.grid_size - 1) if self.grid_size > 1 else 0
        state[6] = self.cargo_depot_pos[1] / (self.grid_size - 1) if self.grid_size > 1 else 0
        
        # === HEDEF BELİRLEME LOJİĞİ ===
        target_x, target_y = 0, 0
        if not self.has_cargo:
            # Kargo yoksa → Kargo deposuna git
            target_x = self.cargo_depot_pos[0] / (self.grid_size - 1) if self.grid_size > 1 else 0
            target_y = self.cargo_depot_pos[1] / (self.grid_size - 1) if self.grid_size > 1 else 0
        else:
            # Kargo varsa → En yakın teslim edilmemiş noktaya git
            min_dist = float('inf')
            closest_point = None
            for i, point in enumerate(self.delivery_points):
                if not self.delivered[i]:
                    # Manhattan distance hesapla
                    dist = abs(self.drone_pos[0] - point[0]) + abs(self.drone_pos[1] - point[1])
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = point
            
            if closest_point is not None:
                target_x = closest_point[0] / (self.grid_size - 1) if self.grid_size > 1 else 0
                target_y = closest_point[1] / (self.grid_size - 1) if self.grid_size > 1 else 0
        
        # === HEDEF BİLGİLERİ ===
        state[7] = target_x
        state[8] = target_y
        
        # === HEDEFE MESAFE (Normalize edilmiş) ===
        distance_to_target = abs(self.drone_pos[0] - target_x * (self.grid_size - 1)) + \
                           abs(self.drone_pos[1] - target_y * (self.grid_size - 1))
        state[9] = distance_to_target / (2 * self.grid_size)  # Maksimum mesafe ile normalize et
        
        # === TESLİMAT DURUMU ===
        state[10] = sum(self.delivered) / len(self.delivered) if self.delivered else 0
        
        # === ZAMAN BİLGİSİ ===
        state[11] = (self.max_steps - self.steps) / self.max_steps  # Kalan adım oranı
        
        return state

    def step(self, action):
        """
        Drone'a verilen eylemi uygular ve ortamı bir adım ilerletir.
        Args:
            action (int):
                0: Aşağı, 1: Sağa, 2: Yukarı, 3: Sola
                4: Kargo Al/Bırak (Yerdeyken kargo al veya teslim et)
                5: Kalk/İn (Take off/landing, uçuş durumunu değiştirir)
        Returns:
            tuple: (next_state, reward, done, info)
                next_state: Yeni durumun hashlenmiş temsili
                reward: Bu adımda alınan ödül/ceza
                done: Senaryo tamamlandı mı?
                info: Ek bilgi (ör. neden bitti, hangi eylem yapıldı)
        Drone kargo almak ve bırakmak için landing durumunda olmalı. Havadayken kargo bırakılamaz alınamaz.
        """
        if self.done:
            return self.get_state(), 0, True, {"info": "Senaryo zaten tamamlanmış."}
        
        # Başlangıç durumu
        old_pos = self.drone_pos.copy()
        reward = 0
        info = {}
        action_emojis = {
            0: '⬇️',  # Aşağı
            1: '➡️',  # Sağa
            2: '⬆️',  # Yukarı
            3: '⬅️',  # Sola
            4: '📦',  # Kargo Al/Bırak
            5: '🛫/🛬',  # Kalk/İn
        }
        action_names = {
            0: 'Aşağı hareket',
            1: 'Sağa hareket',
            2: 'Yukarı hareket',
            3: 'Sola hareket',
            4: 'Kargo Al/Bırak',
            5: 'Kalk/İn',
        }
        # --- Eylem tipine göre ödül/ceza ---
        if action <= 3:  # Hareket eylemleri
            if not self.is_flying:
                reward -= 0.5  # Daha az ceza, öğrenmeyi kolaylaştır
                info["action"] = f"{action_emojis[action]} {action_names[action]} (action={action}) | Drone yerdeyken hareket edemez! Önce kalkış yapın."
            else:
                if action == 0:
                    self.drone_pos[0] = min(self.drone_pos[0] + 1, self.grid_size - 1)
                elif action == 1:
                    self.drone_pos[1] = min(self.drone_pos[1] + 1, self.grid_size - 1)
                elif action == 2:
                    self.drone_pos[0] = max(self.drone_pos[0] - 1, 0)
                elif action == 3:
                    self.drone_pos[1] = max(self.drone_pos[1] - 1, 0)
                if np.array_equal(old_pos, self.drone_pos):
                    reward -= 1  # Daha az ceza
                    info["action"] = f"{action_emojis[action]} {action_names[action]} (action={action}) | Hareket edilemedi."
                else:
                    reward -= 0.2  # Küçük hareket cezası
                    self.battery -= self.move_battery_cost
                    info["action"] = f"{action_emojis[action]} {action_names[action]} (action={action})"
        elif action == 4:  # Kargo Al/Bırak
            if self.is_flying:
                reward -= 2  # Daha az ceza
                info["action"] = f"{action_emojis[action]} {action_names[action]} (action={action}) | Drone havadayken kargo alınamaz/bırakılamaz! Önce iniş yapın."
            else:
                if np.array_equal(self.drone_pos, self.cargo_depot_pos) and not self.has_cargo:
                    self.has_cargo = True
                    reward += 200  # Kargo alma ödülü artırıldı
                    info["action"] = f"{action_emojis[action]} Kargo alındı (action={action})"
                elif self.has_cargo:
                    delivered_any = False
                    for i, delivery_point in enumerate(self.delivery_points):
                        if np.array_equal(self.drone_pos, delivery_point) and not self.delivered[i]:
                            self.delivered[i] = True
                            self.has_cargo = False
                            reward += 500  # Teslimat ödülü artırıldı
                            info["action"] = f"{action_emojis[action]} {i+1}. teslimat tamamlandı (action={action})"
                            delivered_any = True
                            break
                    if not delivered_any:
                        reward -= 5  # Yanlış yerde teslimat cezası
                        info["action"] = f"{action_emojis[action]} Yanlış yerde teslimat (action={action})"
                else:
                    reward -= 5  # Yanlış yerde kargo alma/bırakma cezası
                    info["action"] = f"{action_emojis[action]} Burada kargo alınamaz/bırakılamaz (action={action})"
        elif action == 5:  # Kalk/İn
            if not self.is_flying:
                self.is_flying = True
                self.landing_state = "taking_off"
                self.landing_animation_step = 0
                reward -= 0.5  # Daha az ceza
                info["action"] = f"🛫 Kalkış (action={action})"
                self.battery -= self.takeoff_battery_cost
            else:
                self.is_flying = False
                self.landing_state = "landing"
                self.landing_animation_step = 0
                reward -= 0.5  # Daha az ceza
                info["action"] = f"🛬 İniş (action={action})"
                self.battery -= self.landing_battery_cost

        # --- Hedefe yaklaşma/uzaklaşma ödül/ceza ---
        target_pos = None
        if not self.has_cargo and not all(self.delivered):
            target_pos = self.cargo_depot_pos
            # Kargo almak için ek motivasyon: kargo deposuna yaklaştıkça bonus
            depot_dist = np.sum(np.abs(self.drone_pos - self.cargo_depot_pos))
            if depot_dist <= 1 and not self.is_flying:
                reward += 15  # Kargo deposuna yakın ve yerdeyse bonus
        elif self.has_cargo:
            min_dist = float('inf')
            for i, point in enumerate(self.delivery_points):
                if not self.delivered[i]:
                    dist = np.sum(np.abs(self.drone_pos - point))
                    if dist < min_dist:
                        min_dist = dist
                        target_pos = point
                        
        if target_pos is not None:
            old_dist = np.sum(np.abs(old_pos - target_pos))
            new_dist = np.sum(np.abs(self.drone_pos - target_pos))
            if self.is_flying and new_dist < old_dist:
                reward += 10  # Hedefe yaklaşma ödülü artırıldı
            elif self.is_flying and new_dist > old_dist:
                reward -= 1  # Hedeften uzaklaşma cezası azaltıldı
            if np.array_equal(self.drone_pos, target_pos):
                if not self.is_flying and action == 4:
                    reward += 10  # Doğru yerde doğru eylem bonusu
                elif self.is_flying and action == 5:
                    reward += 5  # Doğru yerde iniş bonusu

        # --- Batarya kontrolü ---
        if self.battery <= 0:
            reward -= 50  # Batarya biterse ceza azaltıldı
            self.battery = 0
            self.done = True
            info["done_reason"] = "Batarya bitti"

        # --- Adım sınırı ---
        self.steps += 1
        if self.steps >= self.max_steps:
            reward -= 20  # Maksimum adım cezası azaltıldı
            self.done = True
            info["done_reason"] = "Maksimum adım sayısına ulaşıldı"

        # --- Tüm teslimatlar tamamlandıysa ---
        if all(self.delivered):
            remaining_battery_bonus = self.battery
            reward += 500 + remaining_battery_bonus  # Çok büyük ödül ve kalan batarya bonusu
            self.done = True
            info["done_reason"] = f"Tüm teslimatlar tamamlandı! Kalan batarya: %{self.battery}"

        # İniş/kalkış animasyon durumlarını güncelle
        # Bu adımlar, görsel arayüzde animasyonun düzgün çalışmasını sağlar.
        if self.landing_state == "taking_off":
            self.landing_animation_step += 1
            if self.landing_animation_step >= 3:  # 3 adımda tamamlanan kalkış animasyonu
                self.landing_state = "flying"
        elif self.landing_state == "landing":
            self.landing_animation_step += 1
            if self.landing_animation_step >= 3:  # 3 adımda tamamlanan iniş animasyonu
                self.landing_state = "landed"
        
        self.last_reward = reward  # Son ödül bilgisini güncelle
        self.total_reward += reward  # Toplam ödülü güncelle
        # Son aksiyon bilgisini ortamda sakla
        self.last_action_info = info.get("action", "-")
        return self.get_state(), reward, self.done, info # Yeni durum, ödül, bölüm durumu ve ek bilgiyi döndür.
# =====================
# DEEP Q-NETWORK (DQN) NEURAL NETWORK
# =====================
class DQN(nn.Module):
    """
    ========================================================================
    DEEP Q-NETWORK SINIR AĞI (PYTORCH IMPLEMENTATION)
    ========================================================================
    DQN algoritmasının kalbi olan neural network yapısı.
    
    Mimari:
    - Input Layer: 12 nöron (state vector boyutu)
    - Hidden Layer 1: 128 nöron + ReLU activation
    - Output Layer: 6 nöron (action space boyutu)
    
    Özellikler:
    - Dropout: %20 overfitting önleme
    - ReLU Activation: Non-linearity için
    
    Bu ağ, state'i alır ve her action için Q-value tahmin eder.
    En yüksek Q-value'ya sahip action seçilir (greedy policy).
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        """
        Neural network katmanlarını tanımla
        
        Args:
            state_size (int): Input boyutu (12)
            action_size (int): Output boyutu (6) 
            hidden_size (int): Hidden layer boyutu (128)
        """
        super(DQN, self).__init__()
        
        # === KATMAN TANIMLARI ===
        self.fc1 = nn.Linear(state_size, hidden_size)      # İlk gizli katman
        self.fc2 = nn.Linear(hidden_size, hidden_size)     # İkinci gizli katman
        self.fc3 = nn.Linear(hidden_size, action_size)     # Çıkış katmanı
        
        # === REGULARİZASYON ===
        self.dropout = nn.Dropout(0.2)  # %20 dropout ile overfitting önleme
        
    def forward(self, x):
        """
        İleri geçiş (forward pass) - state'den Q-values'a
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Q-values for all actions
        """
        # Input → Hidden Layer 1 (ReLU activation)
        x = F.relu(self.fc1(x))
        
        # Dropout uygula (sadece training sırasında)
        x = self.dropout(x)
        
        # Hidden Layer 1 → Hidden Layer 2 (ReLU activation)
        x = F.relu(self.fc2(x))
        
        # Hidden Layer 2 → Output (Linear, no activation)
        x = self.fc3(x)
        
        return x  # Q-values for all 6 actions

# =====================
# DQN AGENT (REINFORCEMENT LEARNING AGENT)
# =====================
class DQNAgent:
    """
    ========================================================================
    DEEP Q-NETWORK AGENT - PEKİŞTİRMELİ ÖĞRENME AJANI
    ========================================================================
    DQN algoritmasının tam implementasyonu. Bu sınıf:
    
    🧠 Öğrenme Mekanizmaları:
    - Experience Replay: Geçmiş deneyimlerden batch öğrenme
    - Target Network: Kararlı eğitim için ayrı hedef ağı
    - Epsilon-Greedy: Exploration vs Exploitation dengesi
    
    📊 Hiperparametreler:
    - Learning Rate: 0.001 (Adam optimizer)
    - Gamma: 0.99 (discount factor)
    - Epsilon: 1.0 → 0.01 (exploration decay)
    - Batch Size: 32 (mini-batch learning)
    - Buffer Size: 10,000 (experience replay)
    
    🎯 Algoritma Adımları:
    1. Action selection (epsilon-greedy)
    2. Experience storage (replay buffer)
    3. Batch learning (Q-learning update)
    4. Target network update (stability)
    """
    def __init__(self, env, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.998, 
                 min_epsilon=0.01, buffer_size=20000, batch_size=32, target_update=250):
        """
        DQN Agent'ı başlat
        
        Args:
            env: Drone delivery environment
            lr (float): Learning rate (0.001)
            gamma (float): Discount factor (0.99)
            epsilon (float): Initial exploration rate (1.0)
            epsilon_decay (float): Exploration decay rate (0.995)
            min_epsilon (float): Minimum exploration rate (0.01)
            buffer_size (int): Experience replay buffer size (10,000)
            batch_size (int): Mini-batch size for learning (32)
            target_update (int): Target network update frequency (100)
        """
        # === TEMEL PARAMETRELER ===
        self.env = env
        self.state_size = 12                    # Optimize edilmiş state size
        self.action_size = env.action_space_n   # 6 action
        
        # === ÖĞRENME HİPERPARAMETRELERİ ===
        self.lr = lr                    # Learning rate
        self.gamma = gamma              # Discount factor (gelecek ödül oranı)
        self.epsilon = epsilon          # Exploration rate (keşif oranı)
        self.epsilon_decay = epsilon_decay  # Epsilon azalma oranı (biraz daha yavaş decay)
        self.min_epsilon = min_epsilon  # Minimum epsilon
        self.batch_size = batch_size    # Mini-batch boyutu
        self.target_update = target_update  # Target network güncelleme sıklığı
        
        # === DEVICE SEÇİMİ (GPU/CPU) ===
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")
        
        # === NEURAL NETWORKS ===
        # Ana network: Eğitilen, gradient alan network
        self.q_network = DQN(self.state_size, self.action_size).to(self.device)
        # Target network: Kararlı target Q-values için ayrı network
        self.target_network = DQN(self.state_size, self.action_size).to(self.device)
        
        # === OPTIMIZER ===
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # === EXPERIENCE REPLAY BUFFER ===
        # Geçmiş deneyimleri saklar: (state, action, reward, next_state, done)
        self.memory = deque(maxlen=buffer_size)
        
        # === SAYAÇLAR ===
        self.step_count = 0     # Toplam adım sayısı
        self.update_count = 0   # Güncelleme sayısı
        
        # İlk target network güncellemesi (ağırlıkları kopyala)
        self.update_target_network()
        
    def select_action(self, state, training=True):
        """
        ===============================================
        EPSILON-GREEDY ACTION SELECTION
        ===============================================
        DQN'in kalbi: Exploration vs Exploitation dengesi
        
        Epsilon-Greedy Stratejisi:
        - Epsilon olasılıkla: Rastgele action seç (EXPLORATION) 🎲
        - (1-Epsilon) olasılıkla: En iyi Q-value'lu action seç (EXPLOITATION) 🎯
        
        Eğitim ilerledikçe epsilon azalır:
        1.0 → 0.01 (keşiften istismara geçiş)
        
        Args:
            state (np.ndarray): Mevcut durum vektörü
            training (bool): Eğitim modu mu? (epsilon kullanılsın mı?)
            
        Returns:
            int: Seçilen action (0-5 arası)
        """
        if training and random.random() < self.epsilon:
            # EXPLORATION: Rastgele action seç
            return random.randint(0, self.action_size - 1)
        else:
            # EXPLOITATION: En iyi Q-value'lu action seç
            with torch.no_grad():  # Gradient hesaplama, sadece inference
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()  # En yüksek Q-value'nun indeksi
    
    def remember(self, state, action, reward, next_state, done):
        """
        ===============================================
        EXPERIENCE REPLAY BUFFER'A DENEYİM EKLE
        ===============================================
        Her adımdan sonra deneyimi buffer'a kaydeder.
        Bu deneyimler daha sonra batch halinde öğrenme için kullanılır.
        
        Experience Tuple: (s, a, r, s', done)
        - s: Mevcut state
        - a: Yapılan action  
        - r: Alınan reward
        - s': Sonraki state
        - done: Episode bitmiş mi?
        
        Args:
            state: Mevcut durum
            action: Yapılan eylem
            reward: Alınan ödül
            next_state: Sonraki durum
            done: Episode tamamlandı mı?
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self, state, action, reward, next_state, done):
        """
        ===============================================
        ÖĞRENME DÖNGÜSÜ (MAIN LEARNING LOOP)
        ===============================================
        Her adımdan sonra çağrılan ana öğrenme fonksiyonu:
        
        1. Deneyimi buffer'a kaydet
        2. Yeterli deneyim varsa batch öğrenme yap
        3. Target network'ü belirli aralıklarla güncelle
        
        Bu fonksiyon DQN'in öğrenme hızını kontrol eder.
        
        Args:
            state: Önceki durum
            action: Yapılan eylem
            reward: Alınan ödül
            next_state: Yeni durum
            done: Episode bitti mi?
        """
        # 1. Deneyimi buffer'a kaydet
        self.remember(state, action, reward, next_state, done)
        
        self.step_count += 1
        
        # 2. Her 4 adımda bir batch öğrenme yap (computational efficiency)
        if len(self.memory) > self.batch_size and self.step_count % 4 == 0:
            self.replay()
            
        # 3. Target network'ü belirli aralıklarla güncelle (stability)
        if self.step_count % self.target_update == 0:
            self.update_target_network()
    
    def replay(self):
        """Experience replay ile batch öğrenme"""
        if len(self.memory) < self.batch_size:
            return
            
        # Random batch seç
        batch = random.sample(self.memory, self.batch_size)
        
        # Her state'in aynı boyutta olduğundan emin ol
        states_list = []
        actions_list = []
        rewards_list = []
        next_states_list = []
        dones_list = []
        
        for experience in batch:
            state, action, reward, next_state, done = experience
            
            # State'lerin numpy array olduğundan ve doğru boyutta olduğundan emin ol
            if isinstance(state, np.ndarray) and state.shape == (12,):  # 12 boyutlu state
                states_list.append(state)
            else:
                # Eğer state doğru formatta değilse, varsayılan state oluştur
                states_list.append(np.zeros(12))
                
            if isinstance(next_state, np.ndarray) and next_state.shape == (12,):
                next_states_list.append(next_state)
            else:
                next_states_list.append(np.zeros(12))
                
            actions_list.append(action)
            rewards_list.append(reward)
            dones_list.append(done)
        
        # Numpy array'lere dönüştür
        states = np.array(states_list, dtype=np.float32)
        actions = np.array(actions_list, dtype=np.int64)
        rewards = np.array(rewards_list, dtype=np.float32)
        next_states = np.array(next_states_list, dtype=np.float32)
        dones = np.array(dones_list, dtype=bool)
        
        # Tensor'lara dönüştür
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values (Target network ile)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Loss hesapla ve backpropagation
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.update_count += 1
    
    def update_target_network(self):
        """Target network'u main network ile güncelle"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Epsilon'u azalt"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filename):
        """Modeli kaydet"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'update_count': self.update_count
        }, filename)
    
    def load_model(self, filename):
        """Modeli yükle"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.min_epsilon)
        self.step_count = checkpoint.get('step_count', 0)
        self.update_count = checkpoint.get('update_count', 0)

# =====================
# Eğitim Thread'i (PyQt5)
# =====================
class TrainingThread(QThread): # PyQt5 QThread sınıfından miras alır, böylece arayüz donmadan eğitim yapılabilir.
    progress = pyqtSignal(int, float, float, float)  # episode, reward, steps, epsilon -> Eğitim ilerlemesini bildiren sinyal.
    finished = pyqtSignal(list, list) # Eğitim bittiğinde ödül ve adım listelerini gönderen sinyal.
    state_update = pyqtSignal() # Ortam durumunun güncellenmesi gerektiğini bildiren sinyal (görsel arayüz için).
    def __init__(self, env, agent, episodes, update_interval=10, mode="fast", delay=0.1): # "ansi" -> "fast"
        super().__init__()
        self.env = env # Eğitim ortamı.
        self.agent = agent # Eğitilecek ajan.
        self.episodes = episodes # Toplam eğitim bölümü sayısı.
        self.running = True # Eğitimin devam edip etmediğini kontrol eden bayrak.
        self.update_interval = update_interval # fast modunda ne sıklıkta arayüzün güncelleneceği.
        self.mode = mode  # 'human' (canlı izleme) veya 'fast' (hızlı eğitim).
        self.delay = delay  # 'human' modunda adımlar arası gecikme (saniye).
    def run(self):
        # Eğitim döngüsü (her episode için)
        rewards_per_episode = [] # Her bölümdeki toplam ödülü saklar.
        steps_per_episode = [] # Her bölümdeki adım sayısını saklar.
        for episode in range(self.episodes):
            if not self.running: # Eğer durdurma sinyali geldiyse eğitimi sonlandır.
                break
            state = self.env.reset() # Ortamı sıfırla.
            total_reward = 0 # Bu bölümdeki toplam ödül.
            done = False # Bölümün bitip bitmediği.
            step_counter = 0 # Bu bölümdeki adım sayısı.
            self.state_update.emit() # Arayüzü güncelle.
            while not done and self.running: # Bölüm bitene kadar veya durdurma sinyali gelene kadar devam et.
                action = self.agent.select_action(state, training=True) # Ajan bir eylem seçer.
                next_state, reward, done, info = self.env.step(action) # Ortamda eylemi uygula.
                self.agent.learn(state, action, reward, next_state, done) # Ajan öğrenir.
                state = next_state # Durumu güncelle.
                total_reward += reward # Toplam ödülü güncelle.
                step_counter += 1
                if self.mode == "human": # Eğer 'human' modundaysa
                    self.state_update.emit() # Arayüzü her adımda güncelle.
                    QThread.msleep(int(self.delay * 1000)) # Belirlenen süre kadar bekle.
                elif self.mode == "fast" and step_counter % self.update_interval == 0: # Eğer 'fast' modundaysa ve belirli aralıklarla # "ansi" -> "fast"
                    self.state_update.emit() # Arayüzü güncelle.
            rewards_per_episode.append(total_reward) # Bölüm ödülünü listeye ekle.
            steps_per_episode.append(self.env.steps) # Bölüm adım sayısını listeye ekle.
            self.agent.decay_epsilon() # Epsilon değerini azalt.
            self.state_update.emit() # Arayüzü güncelle.
            self.progress.emit(episode+1, total_reward, self.env.steps, self.agent.epsilon) # İlerleme sinyalini gönder.
        self.finished.emit(rewards_per_episode, steps_per_episode) # Eğitim bitti sinyalini gönder.
    def stop(self):
        # Eğitimi durdurmak için kullanılır.
        self.running = False

# =====================
# Grid ve Bilgi Paneli (PyQt5)
# =====================
class GridWidget(QWidget): # Ortamın grid yapısını görselleştiren widget.
    def __init__(self, env, parent=None):
        super().__init__(parent)
        self.env = env # Görselleştirilecek ortam.
        self.cell_size = 80 # Her bir grid hücresinin piksel boyutu.
        self.setMinimumSize(env.grid_size * self.cell_size, env.grid_size * self.cell_size)
        # Renkler ve görsel ayarlar
        self.colors = {
            'background': Qt.white,
            'grid': Qt.lightGray,
            'drone': Qt.blue,
            'drone_landed': QColor(100, 100, 180), # İniş yapmış drone rengi.
            'cargo_depot': Qt.green, # Kargo deposu rengi.
            'delivery_point': Qt.red, # Teslimat noktası rengi.
            'cargo': Qt.green, # Kargo rengi.
            'shadow': QColor(100, 100, 100, 80) # Drone uçarkenki gölge rengi.
        }
    def paintEvent(self, event):
        # Grid ve tüm nesneleri çiz
        # Bu fonksiyon, widget her yeniden çizildiğinde çağrılır.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing) # Daha pürüzsüz çizimler için.
        painter.fillRect(self.rect(), self.colors['background']) # Arka planı boya.
        # Ortalamak için offset hesapla
        grid_pixel_size = self.env.grid_size * self.cell_size
        x_offset = (self.width() - grid_pixel_size) // 2
        y_offset = (self.height() - grid_pixel_size) // 2
        # Grid çizgileri
        painter.setPen(QPen(self.colors['grid'], 1))
        for i in range(self.env.grid_size + 1):
            painter.drawLine(x_offset, y_offset + i * self.cell_size, x_offset + self.env.grid_size * self.cell_size, y_offset + i * self.cell_size)
            painter.drawLine(x_offset + i * self.cell_size, y_offset, x_offset + i * self.cell_size, y_offset + self.env.grid_size * self.cell_size)
        # Kargo deposu çizimi
        depot_x = x_offset + self.env.cargo_depot_pos[1] * self.cell_size + self.cell_size // 2
        depot_y = y_offset + self.env.cargo_depot_pos[0] * self.cell_size + self.cell_size // 2
        painter.setBrush(QBrush(self.colors['cargo_depot']))
        painter.setPen(Qt.NoPen) # Kenar çizgisi olmasın.
        painter.drawEllipse(depot_x - self.cell_size // 3, depot_y - self.cell_size // 3, 2 * self.cell_size // 3, 2 * self.cell_size // 3)
        # Teslimat noktaları çizimi
        painter.setBrush(QBrush(self.colors['delivery_point']))
        for i, point in enumerate(self.env.delivery_points):
            if i < len(self.env.delivered) and not self.env.delivered[i]: # Henüz teslim edilmemişse çiz.
                x = x_offset + point[1] * self.cell_size + self.cell_size // 2
                y = y_offset + point[0] * self.cell_size + self.cell_size // 2
                painter.drawEllipse(x - self.cell_size // 4, y - self.cell_size // 4, self.cell_size // 2, self.cell_size // 2)
                painter.setPen(Qt.black) # Teslimat noktası numarasını yazmak için.
                painter.setFont(QFont('Arial', 10))
                painter.drawText(x - 5, y + 5, str(i + 1)) # Teslimat noktası numarasını yaz.
                painter.setPen(Qt.NoPen)
        # Drone çizimi
        drone_x = x_offset + self.env.drone_pos[1] * self.cell_size + self.cell_size // 2
        drone_y = y_offset + self.env.drone_pos[0] * self.cell_size + self.cell_size // 2
        if self.env.is_flying: # Drone uçuyorsa
            # Gölge efekti
            painter.setBrush(QBrush(self.colors['shadow']))
            painter.drawEllipse(drone_x - self.cell_size // 6, drone_y + self.cell_size // 4, self.cell_size // 3, self.cell_size // 8)
            height_offset = 0 # Yükseklik ofseti (animasyon için).
            if self.env.landing_state == "taking_off": # Kalkış animasyonu
                height_offset = -5 * self.env.landing_animation_step
            elif self.env.landing_state == "landing": # İniş animasyonu
                height_offset = -15 + 5 * self.env.landing_animation_step
            elif self.env.landing_state == "flying": # Normal uçuş
                height_offset = -15
            drone_y += height_offset # Drone'un dikey konumunu ayarla.
            painter.setBrush(QBrush(self.colors['drone'])) # Uçan drone rengi.
        else: # Drone yerdeyse
            painter.setBrush(QBrush(self.colors['drone_landed'])) # İniş yapmış drone rengi.
        # Drone gövdesi
        painter.drawEllipse(drone_x - self.cell_size // 4, drone_y - self.cell_size // 4, self.cell_size // 2, self.cell_size // 2)
        # Pervaneler
        propeller_size = self.cell_size // 8
        if self.env.is_flying: # Uçarken pervaneler daha büyük görünebilir.
            propeller_size = self.cell_size // 6
        painter.setBrush(QBrush(Qt.black)) # Pervane rengi.
        # Sol üst
        painter.drawEllipse(drone_x - propeller_size - propeller_size//2, drone_y - propeller_size - propeller_size//2, propeller_size, propeller_size)
        # Sağ üst
        painter.drawEllipse(drone_x + propeller_size - propeller_size//2, drone_y - propeller_size - propeller_size//2, propeller_size, propeller_size)
        # Sol alt
        painter.drawEllipse(drone_x - propeller_size - propeller_size//2, drone_y + propeller_size - propeller_size//2, propeller_size, propeller_size)
        # Sağ alt
        painter.drawEllipse(drone_x + propeller_size - propeller_size//2, drone_y + propeller_size - propeller_size//2, propeller_size, propeller_size)
        # Kargo çizimi
        if self.env.has_cargo: # Eğer drone kargo taşıyorsa
            painter.setBrush(QBrush(self.colors['cargo'])) # Kargo rengi.
            painter.drawRect(drone_x - self.cell_size // 8, drone_y - self.cell_size // 8, self.cell_size // 4, self.cell_size // 4)
        # Batarya göstergesi
        painter.setPen(Qt.black)
        painter.setFont(QFont('Arial', 10))
        painter.drawText(drone_x - 20, drone_y - 30, f"🔋: {self.env.battery}%") # Drone üzerinde batarya seviyesini göster.

class InfoPanelWidget(QWidget): # Ortam ve eğitim bilgilerini gösteren widget.
    def __init__(self, env, parent=None):
        super().__init__(parent)
        self.env = env # Bilgileri gösterilecek ortam.
        layout = QVBoxLayout()
        # Bilgi paneli için GroupBox
        info_group = QGroupBox("ℹ️ Durum Bilgileri")
        info_layout = QVBoxLayout()
        self.battery_label = QLabel() # Batarya bilgisi etiketi.
        self.cargo_label = QLabel() # Kargo durumu etiketi.
        self.delivery_label = QLabel() # Teslimat durumu etiketi.
        self.steps_label = QLabel() # Adım sayısı etiketi.
        self.reward_label = QLabel()  # Son ödül etiketi
        self.total_reward_label = QLabel()  # Toplam ödül etiketi
        self.last_action_label = QLabel()  # Son aksiyon etiketi
        self.training_progress_label = QLabel()  # Eğitim ilerlemesi etiketi
        info_layout.addWidget(self.battery_label)
        info_layout.addWidget(self.cargo_label)
        info_layout.addWidget(self.delivery_label)
        info_layout.addWidget(self.steps_label)
        info_layout.addWidget(self.reward_label)  # Son ödül panelde göster
        info_layout.addWidget(self.total_reward_label)  # Toplam ödül panelde göster
        info_layout.addWidget(self.last_action_label)  # Son aksiyon panelde göster
        info_layout.addWidget(self.training_progress_label)  # Durum bilgisine eklendi
        info_group.setLayout(info_layout)
        self.status_label = QLabel() # Genel durum mesajları için etiket.
        # Ana layout
        layout.addWidget(info_group)
        layout.addWidget(self.status_label)
        layout.addStretch() # Widget'ları yukarıya iter.
        self.setLayout(layout)
        self.update_info() # Bilgileri ilk kez güncelle.
    def update_info(self):
        # Paneldeki tüm bilgileri günceller.
        self.battery_label.setText(f"🔋 Batarya: %{self.env.battery}")
        # Kargo etiketi: taşınıyorsa kalın yeşil
        if self.env.has_cargo:
            self.cargo_label.setText("📦 Kargo: <span style='color:#1ca81c; font-weight:bold;'>Taşınıyor</span>")
            self.cargo_label.setTextFormat(Qt.RichText) # HTML formatında metin.
            self.cargo_label.setStyleSheet("")
        else:
            self.cargo_label.setText("📦 Kargo: Yok")
            self.cargo_label.setTextFormat(Qt.AutoText)
            self.cargo_label.setStyleSheet("")
        # Teslimat etiketi: teslim edilen sayı yeşil ve kalın
        delivered_count = sum(self.env.delivered) # Teslim edilen paket sayısı.
        total = len(self.env.delivery_points) # Toplam teslimat noktası sayısı.
        if delivered_count > 0:
            self.delivery_label.setText(f"🎯 Teslimatlar: <span style='color:#1ca81c; font-weight:bold;'>{delivered_count}</span>/{total}")
            self.delivery_label.setTextFormat(Qt.RichText)
        else:
            self.delivery_label.setText(f"🎯 Teslimatlar: 0/{total}")
            self.delivery_label.setTextFormat(Qt.AutoText)
        self.steps_label.setText(f"👣 Adım: {self.env.steps}")
        self.reward_label.setText(f"🏅 Son Ödül: {self.env.last_reward:.2f}")  # Son ödül gösterimi
        self.total_reward_label.setText(f"🥇 Toplam Ödül: {self.env.total_reward:.2f}")  # Toplam ödül gösterimi
        # Son aksiyon bilgisini grup kutusunda göster
        if hasattr(self.env, 'last_action_info') and self.env.last_action_info:
            self.last_action_label.setText(f"🔄 Son Aksiyon: {self.env.last_action_info}")
        else:
            self.last_action_label.setText("🔄 Son Aksiyon: -")
        # Eğitim ilerlemesi sadece eğitim sırasında gösterilecek, aksi halde gizle
        if not self.training_progress_label.text(): # Eğer eğitim ilerleme metni boşsa
            self.training_progress_label.setVisible(False) # Etiketi gizle.
        else:
            self.training_progress_label.setVisible(True) # Etiketi göster.
    def set_status(self, status):
        # Genel durum mesajını ayarlar.
        self.status_label.setText(status)
    def set_training_progress(self, episode, total_episodes, reward, steps, epsilon):
        # Eğitim ilerleme bilgisini ayarlar.
        self.training_progress_label.setText(f"📈 Episode: {episode}/{total_episodes} | Ödül: {reward:.2f} | Adım: {steps}")
        self.training_progress_label.setVisible(True) # Etiketi görünür yap.
    def clear_training_progress(self):
        # Eğitim ilerleme bilgisini temizler ve gizler.
        self.training_progress_label.setText("")
        self.training_progress_label.setVisible(False)

# =====================
# Neural Network Görselleştirme Widget'ı
# =====================
class NeuralNetworkWidget(QWidget):
    """
    Basit ve anlaşılır DQN neural network görselleştirmesi
    - 3 katman: Input → Hidden → Output
    - Her katmanda 5-6 nöron
    - Renkli aktivasyon gösterimi
    """
    def __init__(self, agent, parent=None):
        super().__init__(parent)
        self.agent = agent
        self.setMinimumSize(300, 200)
        self.setMaximumSize(300, 220)
        
        # Basit katman konfigürasyonu
        self.layer_names = ["Input", "Hidden", "Output"]
        self.layer_sizes = [5, 6, 6]  # Görselleştirme için basit
        self.action_names = ["↓", "→", "↑", "←", "📦", "🛫"]
        
        # Aktivasyon verileri
        self.activations = [np.random.random(size) * 0.3 for size in self.layer_sizes]
        self.last_action = 0
        
        # Basit renkler
        self.bg_color = QColor(240, 240, 240)
        self.neuron_colors = {
            'inactive': QColor(200, 200, 200),
            'low': QColor(150, 200, 255),
            'medium': QColor(100, 150, 255),
            'high': QColor(255, 100, 100),
            'selected': QColor(50, 255, 50)
        }
        self.text_color = Qt.black
        self.connection_color = QColor(150, 150, 150, 100)
        
    def update_activations(self, state, q_values=None, selected_action=None):
        """Agent'tan gelen state ve Q-values ile aktivasyonları güncelle"""
        try:
            if isinstance(state, np.ndarray) and len(state) >= 5:
                # Input layer - state'in önemli parçalarını seç
                self.activations[0] = np.array([
                    state[0] if len(state) > 0 else 0,  # drone_x
                    state[1] if len(state) > 1 else 0,  # drone_y
                    state[2] if len(state) > 2 else 0,  # has_cargo
                    state[3] if len(state) > 3 else 0,  # is_flying
                    state[4] if len(state) > 4 else 0   # battery
                ])
                
                # Hidden layer - simulated values
                input_avg = np.mean(self.activations[0])
                self.activations[1] = np.random.normal(input_avg, 0.2, 6)
                self.activations[1] = np.clip(self.activations[1], 0, 1)
            
            if q_values is not None and len(q_values) >= 6:
                # Output layer - Q-values normalize
                q_min, q_max = np.min(q_values), np.max(q_values)
                if q_max > q_min:
                    self.activations[2] = (q_values - q_min) / (q_max - q_min)
                else:
                    self.activations[2] = np.ones(6) * 0.5
                    
            if selected_action is not None:
                self.last_action = selected_action
                
        except Exception as e:
            # Fallback random values
            for i in range(len(self.activations)):
                self.activations[i] = np.random.random(self.layer_sizes[i]) * 0.5
        
        self.update()  # Widget'ı yeniden çiz
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)
        
        width = self.width()
        height = self.height()
        
        # Title
        painter.setPen(self.text_color)
        painter.setFont(QFont('Arial', 12, QFont.Bold))
        painter.drawText(10, 20, "🧠 Neural Network")
        
        # Katman pozisyonları
        layer_x = [80, 150, 220]  # 3 katman için sabit pozisyonlar
        start_y = 50
        layer_height = height - 100
        
        # Bağlantıları önce çiz (nöronların altında kalması için)
        for layer_idx in range(len(self.layer_sizes) - 1):
            x1 = layer_x[layer_idx]
            x2 = layer_x[layer_idx + 1]
            
            for i in range(self.layer_sizes[layer_idx]):
                y1 = start_y + (i + 1) * layer_height / (self.layer_sizes[layer_idx] + 1)
                
                for j in range(self.layer_sizes[layer_idx + 1]):
                    y2 = start_y + (j + 1) * layer_height / (self.layer_sizes[layer_idx + 1] + 1)
                    
                    # Bağlantı rengi aktivasyona göre
                    activation = self.activations[layer_idx][i] if i < len(self.activations[layer_idx]) else 0
                    conn_color = QColor(self.connection_color)
                    conn_color.setAlpha(int(50 + activation * 100))
                    
                    painter.setPen(QPen(conn_color, 1))
                    painter.drawLine(int(x1 + 8), int(y1), int(x2 - 8), int(y2))
        
        # Nöronları çiz
        for layer_idx, layer_size in enumerate(self.layer_sizes):
            x = layer_x[layer_idx]
            
            # Katman başlığı
            painter.setPen(self.text_color)
            painter.setFont(QFont('Arial', 10, QFont.Bold))
            painter.drawText(int(x - 20), 40, self.layer_names[layer_idx])
            
            for neuron_idx in range(layer_size):
                y = start_y + (neuron_idx + 1) * layer_height / (layer_size + 1)
                
                # Aktivasyon değeri
                activation = self.activations[layer_idx][neuron_idx] if neuron_idx < len(self.activations[layer_idx]) else 0
                
                # Nöron rengi
                if layer_idx == 2 and neuron_idx == self.last_action:  # Seçilen eylem
                    color = self.neuron_colors['selected']
                elif activation > 0.7:
                    color = self.neuron_colors['high']
                elif activation > 0.4:
                    color = self.neuron_colors['medium']
                elif activation > 0.1:
                    color = self.neuron_colors['low']
                else:
                    color = self.neuron_colors['inactive']
                
                # Nöron çiz
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(Qt.black, 1))
                painter.drawEllipse(int(x - 8), int(y - 8), 16, 16)
                
                # Output layer için eylem etiketleri
                if layer_idx == 2 and neuron_idx < len(self.action_names):
                    painter.setPen(self.text_color)
                    painter.setFont(QFont('Arial', 8))
                    painter.drawText(int(x + 12), int(y + 4), self.action_names[neuron_idx])
                
                # Aktivasyon değerini göster
                if activation > 0.1:
                    painter.setPen(Qt.white)
                    painter.setFont(QFont('Arial', 6))
                    painter.drawText(int(x - 4), int(y + 2), f"{activation:.1f}")
        
        # Legend
        painter.setPen(self.text_color)
        painter.setFont(QFont('Arial', 8))
        painter.drawText(10, height - 15, "🔴 Yüksek  🔵 Orta  ⚪ Düşük")

# =====================
# Ana PyQt5 Arayüzü
# =====================
class DroneDeliverySimulator(QMainWindow): # Ana uygulama penceresi.
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Paket Dağıtım Dronları Simülatörü - DQN") # Pencere başlığı.

        # Emoji ikonu oluşturma
        emoji = "🚁"
        pixmap = QPixmap(64, 64) # İkon boyutu
        pixmap.fill(Qt.transparent) # Şeffaf arka plan
        painter = QPainter(pixmap)
        font = QFont()
        font.setPointSize(48) # Emoji boyutu
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, emoji)
        painter.end()
        self.setWindowIcon(QIcon(pixmap))

        self.resize(1200, 700) # Pencere boyutu.
        self.grid_size = 5 # Başlangıç grid boyutu.
        self.env = DroneDeliveryEnv(grid_size=self.grid_size) # Ortamı oluştur.
        self.agent = DQNAgent(self.env) # DQN ajanı oluştur.
        self.training_thread = None # Eğitim thread'i başlangıçta yok.
        self.sim_speed = 50  # AI ile oyna hız (ms).
        # --- Ana Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget) # Ana widget'ı ayarla.
        main_layout = QHBoxLayout(central_widget) # Ana layout (yatay).
        # --- Sol Panel: Parametreler ve Kontroller ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel) # Sol panel layout'u (dikey).
        # Grid boyutu
        grid_group = QGroupBox("🗺️ Grid Ayarları")
        grid_layout = QHBoxLayout()
        grid_layout.addWidget(QLabel("Grid Boyutu:"))
        self.grid_size_spin = QSpinBox() # Grid boyutu için spin box.
        self.grid_size_spin.setRange(3, 7) # Min ve max grid boyutu.
        self.grid_size_spin.setValue(self.grid_size)
        self.grid_size_spin.valueChanged.connect(self.update_grid_size) # Değer değiştiğinde fonksiyon çağır.
        grid_layout.addWidget(self.grid_size_spin)
        grid_group.setLayout(grid_layout)
        left_layout.addWidget(grid_group)
        
        # DQN parametreleri
        dqn_group = QGroupBox("🤖 DQN Parametreleri")
        dqn_layout = QGridLayout() # Parametreleri grid içinde düzenle.
        dqn_layout.addWidget(QLabel("Learning Rate:"), 0, 0)
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setRange(0.0001, 0.01); self.lr_spin.setSingleStep(0.0001); self.lr_spin.setValue(0.001); self.lr_spin.setDecimals(4)
        dqn_layout.addWidget(self.lr_spin, 0, 1)
        dqn_layout.addWidget(QLabel("Gamma:"), 1, 0)
        self.gamma_spin = QDoubleSpinBox(); self.gamma_spin.setRange(0.1, 0.999); self.gamma_spin.setSingleStep(0.01); self.gamma_spin.setValue(0.99)
        dqn_layout.addWidget(self.gamma_spin, 1, 1)
        dqn_layout.addWidget(QLabel("Epsilon (Keşif Oranı):"), 2, 0)
        self.epsilon_spin = QDoubleSpinBox(); self.epsilon_spin.setRange(0.1, 1.0); self.epsilon_spin.setSingleStep(0.1); self.epsilon_spin.setValue(1.0)
        dqn_layout.addWidget(self.epsilon_spin, 2, 1)
        dqn_layout.addWidget(QLabel("Epsilon Decay:"), 3, 0)
        self.epsilon_decay_spin = QDoubleSpinBox(); self.epsilon_decay_spin.setRange(0.9, 0.9999); self.epsilon_decay_spin.setSingleStep(0.001); self.epsilon_decay_spin.setValue(0.998)
        dqn_layout.addWidget(self.epsilon_decay_spin, 3, 1)
        dqn_layout.addWidget(QLabel("Min Epsilon:"), 4, 0)
        self.min_epsilon_spin = QDoubleSpinBox(); self.min_epsilon_spin.setRange(0.01, 0.5); self.min_epsilon_spin.setSingleStep(0.01); self.min_epsilon_spin.setValue(0.01)
        dqn_layout.addWidget(self.min_epsilon_spin, 4, 1)
        dqn_layout.addWidget(QLabel("Batch Size:"), 5, 0)
        self.batch_size_spin = QSpinBox(); self.batch_size_spin.setRange(16, 128); self.batch_size_spin.setSingleStep(16); self.batch_size_spin.setValue(32)
        dqn_layout.addWidget(self.batch_size_spin, 5, 1)
        dqn_layout.addWidget(QLabel("Eğitim Episodes:"), 6, 0)
        self.episodes_spin = QSpinBox(); self.episodes_spin.setRange(1000, 100000); self.episodes_spin.setSingleStep(1000); self.episodes_spin.setValue(20000) # 20bin episode default
        dqn_layout.addWidget(self.episodes_spin, 6, 1)
        dqn_group.setLayout(dqn_layout)
        left_layout.addWidget(dqn_group)
        # Eğitim hızı (mod ve delay)
        speed_group = QGroupBox("⚡ Eğitim/Simülasyon Hızı")
        speed_layout = QGridLayout()
        speed_layout.addWidget(QLabel("Eğitim Modu:"), 0, 0)
        self.training_mode_combo = QComboBox(); 
        self.training_mode_combo.addItems(["Fast Mode", "Human Mode"]) # Eğitim modu seçenekleri.
        speed_layout.addWidget(self.training_mode_combo, 0, 1)
        speed_layout.addWidget(QLabel("Eğitim Hızı (ms):"), 1, 0) # Canlı mod için eğitim hızı.
        speed_slider_row = QHBoxLayout()
        speed_slider_row.addWidget(QLabel("🏎️"))
        self.training_speed_slider = QSlider(Qt.Horizontal); self.training_speed_slider.setRange(10, 1000); self.training_speed_slider.setValue(100) # Hız ayarı için slider.
        speed_slider_row.addWidget(self.training_speed_slider)
        speed_slider_row.addWidget(QLabel("🐢"))
        speed_layout.addLayout(speed_slider_row, 1, 1)
        speed_layout.addWidget(QLabel("Simülasyon Hızı (ms):"), 2, 0) # AI ile oynama hızı.
        sim_slider_row = QHBoxLayout()
        sim_slider_row.addWidget(QLabel("🏎️"))
        self.sim_speed_slider = QSlider(Qt.Horizontal); self.sim_speed_slider.setRange(10, 1000); self.sim_speed_slider.setValue(self.sim_speed)
        self.sim_speed_slider.valueChanged.connect(self.update_sim_speed) # Değer değiştiğinde fonksiyon çağır.
        sim_slider_row.addWidget(self.sim_speed_slider)
        sim_slider_row.addWidget(QLabel("🐢"))
        speed_layout.addLayout(sim_slider_row, 2, 1)
        speed_group.setLayout(speed_layout)
        left_layout.addWidget(speed_group)
        # Eğitim kontrolleri
        training_group = QGroupBox("🎓 Eğitim")
        training_layout = QVBoxLayout()
        self.train_button = QPushButton("🚀 Eğitimi Başlat"); self.train_button.clicked.connect(self.start_training) # Eğitimi başlat butonu.
        self.stop_button = QPushButton("⏹️ Eğitimi Durdur"); self.stop_button.clicked.connect(self.stop_training); self.stop_button.setEnabled(False) # Eğitimi durdur butonu (başlangıçta pasif).
        self.save_button = QPushButton("💾 Modeli Kaydet"); self.save_button.clicked.connect(self.save_model); self.save_button.setEnabled(False) # Modeli kaydet butonu (başlangıçta pasif).
        self.load_button = QPushButton("📂 Modeli Yükle"); self.load_button.clicked.connect(self.load_model) # Modeli yükle butonu.
        training_layout.addWidget(self.train_button)
        training_layout.addWidget(self.stop_button)
        training_layout.addWidget(self.save_button)
        training_layout.addWidget(self.load_button)
        training_group.setLayout(training_layout)
        left_layout.addWidget(training_group)
        # Oyun kontrolleri
        game_group = QGroupBox("🎮 Oyun Kontrolleri")
        game_layout = QVBoxLayout()
        self.ai_button = QPushButton("🤖 AI ile Oyna"); self.ai_button.clicked.connect(self.play_with_ai); self.ai_button.setEnabled(False) # AI ile oyna butonu (başlangıçta pasif).
        self.human_button = QPushButton("🧑‍💻 İnsan Modu (Manuel Oyna)"); self.human_button.clicked.connect(self.play_human_mode) # Manuel oynama butonu.
        self.stop_game_button = QPushButton("⏹️ Oyunu Durdur"); self.stop_game_button.clicked.connect(self.stop_game); self.stop_game_button.setEnabled(False) # Oyunu durdur butonu (başlangıçta pasif).
        self.reset_button = QPushButton("🔄 Sıfırla"); self.reset_button.clicked.connect(self.reset_env) # Ortamı sıfırla butonu.
        game_layout.addWidget(self.ai_button)
        game_layout.addWidget(self.human_button)
        game_layout.addWidget(self.stop_game_button)
        game_layout.addWidget(self.reset_button)
        game_group.setLayout(game_layout)
        left_layout.addWidget(game_group)
        
        left_layout.addStretch() # Sol paneli yukarı iter.        # --- Sağ Panel: Grid ve Neural Network ---
        right_panel = QWidget()
        right_layout = QHBoxLayout(right_panel) # Sağ panel layout'u (yatay).
        
        # Grid bölümü
        grid_section = QWidget()
        grid_section_layout = QVBoxLayout(grid_section)
        self.grid_widget = GridWidget(self.env) # Grid widget'ını oluştur.
        self.info_panel = InfoPanelWidget(self.env) # Bilgi paneli widget'ını oluştur.
        grid_section_layout.addWidget(self.grid_widget, 7) # Grid widget'ını ekle (daha fazla yer kaplasın).
        grid_section_layout.addWidget(self.info_panel, 3) # Bilgi panelini ekle.
        
        # Neural Network bölümü
        nn_section = QWidget()
        nn_section_layout = QVBoxLayout(nn_section)
        nn_group = QGroupBox("🧠 Neural Network")
        nn_group_layout = QVBoxLayout()
        self.nn_widget = NeuralNetworkWidget(self.agent)
        nn_group_layout.addWidget(self.nn_widget)
        nn_group.setLayout(nn_group_layout)
        nn_section_layout.addWidget(nn_group)
        nn_section_layout.addStretch() # Neural network bölümünü yukarı iter.
        
        # Sağ panele bölümleri ekle
        right_layout.addWidget(grid_section, 3) # Grid bölümü (daha fazla yer)
        right_layout.addWidget(nn_section, 1) # Neural network bölümü (daha az yer)
        # --- Layoutları birleştir ---
        main_layout.addWidget(left_panel, 1) # Sol paneli ana layout'a ekle (daha az yer kaplasın).
        main_layout.addWidget(right_panel, 4) # Sağ paneli ana layout'a ekle (daha fazla yer kaplasın).
        # --- Timer ---
        self.game_timer = QTimer(); self.game_timer.timeout.connect(self.update_game) # Oyun döngüsü için timer.
        self.game_mode = None # Oyun modu (ai, human, None).
        self.model_trained = False # Modelin eğitilip eğitilmediği.
        self.model_loaded = False # Modelin yüklenip yüklenmediği.
        self.update_ui() # Arayüzü ilk kez güncelle.
        self.statusBar().showMessage("Hazır - Eğitim veya AI ile oynamak için modeli eğitin/yükleyin.") # Durum çubuğu mesajı.
        
        # GitHub linki
        github_link_label = QLabel('Coded by <a href="https://github.com/FerhatAkalan">Ferhat Akalan</a>')
        github_link_label.setOpenExternalLinks(True)
        self.statusBar().addPermanentWidget(github_link_label)

    def set_params_enabled(self, enabled: bool):
        # DQN parametrelerinin aktif/pasif durumunu ayarlar.
        self.grid_size_spin.setEnabled(enabled)
        self.lr_spin.setEnabled(enabled)
        self.gamma_spin.setEnabled(enabled)
        self.epsilon_spin.setEnabled(enabled)
        self.epsilon_decay_spin.setEnabled(enabled)
        self.min_epsilon_spin.setEnabled(enabled)
        self.batch_size_spin.setEnabled(enabled)
        self.episodes_spin.setEnabled(enabled)
        self.training_mode_combo.setEnabled(enabled)
        self.training_speed_slider.setEnabled(enabled)
        self.sim_speed_slider.setEnabled(enabled)

    def set_game_buttons_enabled(self, enabled: bool):
        # Oyunla ilgili butonların aktif/pasif durumunu ayarlar.
        self.train_button.setEnabled(enabled)
        self.ai_button.setEnabled(enabled and (self.model_trained or self.model_loaded)) # AI butonu model varsa aktif olur.
        self.human_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled and self.model_trained) # Kaydet butonu model eğitildiyse aktif olur.
        self.load_button.setEnabled(enabled)

    def update_grid_size(self):
        # Grid boyutu değiştiğinde çağrılır.
        self.grid_size = self.grid_size_spin.value()
        self.reset_env() # Ortamı yeni grid boyutuyla sıfırla.
    def update_sim_speed(self):
        # Simülasyon hızı (AI ile oynama hızı) değiştiğinde çağrılır.
        self.sim_speed = self.sim_speed_slider.value()
        if self.game_timer.isActive(): # Eğer oyun zamanlayıcısı aktifse
            self.game_timer.setInterval(self.sim_speed) # Zamanlayıcının aralığını güncelle.
    def reset_env(self):
        # Ortamı ve ajanı sıfırlar.
        if self.game_timer.isActive(): # Eğer oyun zamanlayıcısı aktifse durdur.
            self.game_timer.stop(); self.game_mode = None
        self.env = DroneDeliveryEnv(grid_size=self.grid_size) # Yeni ortam oluştur.
        self.agent = DQNAgent(self.env) # Yeni DQN ajanı oluştur.
        self.grid_widget.env = self.env # Grid widget'ının ortamını güncelle.
        self.info_panel.env = self.env # Bilgi panelinin ortamını güncelle.
        self.model_trained = False # Model eğitilmedi olarak işaretle.
        self.model_loaded = False # Model yüklenmedi olarak işaretle.
        self.update_ui() # Arayüzü güncelle.
        self.set_game_buttons_enabled(True) # Butonları aktif et.
        self.set_params_enabled(True) # Parametreleri aktif et.
        self.info_panel.clear_training_progress() # Eğitim ilerlemesini temizle.
        self.statusBar().showMessage("Ortam sıfırlandı")

    def update_ui(self):
        # Arayüzdeki grid ve bilgi panelini günceller.
        self.grid_widget.update(); self.info_panel.update_info()
        
        # Neural network görselleştirmesini güncelle
        if hasattr(self, 'nn_widget'):
            current_state = self.env.get_state()
            # Agent'tan Q-values al
            with torch.no_grad():
                state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
                q_values = self.agent.q_network(state_tensor).squeeze().numpy()
                selected_action = self.agent.select_action(current_state, training=False)
            self.nn_widget.update_activations(current_state, q_values, selected_action)

    def update_game(self):
        # AI ile oynama modunda oyunun bir adımını günceller.
        if self.game_mode == 'ai':
            if not hasattr(self, 'ai_episode_count'):
                self.ai_episode_count = 1
                self.ai_total_reward = 0
            
            state = self.env.get_state()
            action = self.agent.select_action(state, training=False)
            next_state, reward, done, info = self.env.step(action)
            self.ai_total_reward += reward
            self.update_ui()
            if done:
                delivered = sum(self.env.delivered)
                # Bilgi panelinde ve durum çubuğunda bölüm sonucunu göster.
                self.info_panel.set_status(f"🤖AI Episode: {self.ai_episode_count} | 🎯Teslimat: {delivered}/{len(self.env.delivery_points)} | 🔋Batarya: %{self.env.battery} | 🥇Skor: {self.ai_total_reward:.2f}")
                self.ai_episode_count += 1
                self.ai_total_reward = 0
                # Episode bittiğinde otomatik olarak yeni episode başlat
                self.env.reset()
                self.update_ui()
    def start_training(self):
        # Eğitimi başlatır.
        # Ajan parametrelerini arayüzdeki değerlerle günceller.
        self.agent.lr = self.lr_spin.value()
        self.agent.gamma = self.gamma_spin.value()
        self.agent.epsilon = self.epsilon_spin.value()
        self.agent.epsilon_decay = self.epsilon_decay_spin.value()
        self.agent.min_epsilon = self.min_epsilon_spin.value()
        self.agent.batch_size = self.batch_size_spin.value()
        # Optimizer'ı yeni learning rate ile güncelle
        self.agent.optimizer = optim.Adam(self.agent.q_network.parameters(), lr=self.agent.lr)
        episodes = self.episodes_spin.value() # Eğitim bölümü sayısını al.
        mode_text = self.training_mode_combo.currentText() # Seçilen eğitim modunu al.
        training_mode = "human" if "human" in mode_text.lower() else "fast" # Eğitim modunu belirle.
        delay = self.training_speed_slider.value() / 1000.0 if hasattr(self, 'training_speed_slider') else 0.1 # Canlı mod için gecikme.
        
        self.info_panel.clear_training_progress()  # Eğitim başında ilerleme bilgisini temizle.
        # Buton ve parametrelerin durumunu ayarla (eğitim sırasında çoğu pasif olur).
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_game_button.setEnabled(False)
        self.set_game_buttons_enabled(False)
        self.set_params_enabled(False)
        # ---
        self.env.reset() # Ortamı sıfırla.
        self.training_rewards = []; self.training_steps = [] # Ödül ve adım listelerini sıfırla.
        # Eğitim thread'ini oluştur ve başlat.
        self.training_thread = TrainingThread(self.env, self.agent, episodes, mode=training_mode, delay=delay)
        self.training_thread.progress.connect(self.update_training_progress) # İlerleme sinyaline bağlan.
        self.training_thread.finished.connect(self.training_finished) # Bitiş sinyaline bağlan.
        self.training_thread.state_update.connect(self.update_training_visualization) # Durum güncelleme sinyaline bağlan.
        self.training_thread.start() # Thread'i başlat.
        self.info_panel.set_status("Eğitim devam ediyor...")
        self.statusBar().showMessage(f"Eğitim başladı. Toplam episode: {episodes}")
    def update_training_visualization(self):
        # Eğitim sırasında arayüzü günceller (özellikle canlı modda).
        self.update_ui()
    def update_training_progress(self, episode, reward, steps, epsilon):
        # Eğitim ilerlemesini alır ve arayüzde gösterir.
        self.training_rewards.append(reward)
        self.training_steps.append(steps)
        # Eğitim bilgisi sadece eğitim sırasında gösterilecek
        self.info_panel.set_training_progress(episode, self.episodes_spin.value(), reward, steps, epsilon)
        self.info_panel.update_info() # Bilgi panelini güncelle.
        self.update_ui() # Genel arayüzü güncelle.
    def training_finished(self, rewards, steps):
        # Eğitim bittiğinde çağrılır.
        # Buton ve parametrelerin durumunu eski haline getirir.
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.stop_game_button.setEnabled(False)
        self.set_game_buttons_enabled(True)
        self.info_panel.clear_training_progress()  # Eğitim bitince eğitim bilgisini gizle.
        # Son 100 bölümün ortalama ödül ve adım sayısını hesapla.
        avg_reward = sum(rewards[-100:]) / min(100, len(rewards)) if rewards else 0
        avg_steps = sum(steps[-100:]) / min(100, len(steps)) if steps else 0
        result_message = f"Eğitim tamamlandı!\n\nToplam episode: {len(rewards)}\n"
        result_message += f"Son 100 episode ortalama ödül: {avg_reward:.2f}\n"
        result_message += f"Son 100 episode ortalama adım: {avg_steps:.2f}\n\n"
        result_message += "Şimdi 'AI ile Oyna' butonunu kullanarak eğitilen modeli test edebilirsiniz."
        QMessageBox.information(self, "Eğitim Tamamlandı", result_message) # Bilgilendirme mesajı göster.
        self.statusBar().showMessage(f"Eğitim tamamlandı! Son 100 episode ortalama ödül: {avg_reward:.2f}, adım: {avg_steps:.2f}")
        self.training_thread = None
        self.model_trained = True # Model eğitildi olarak işaretle.
        # self.training_status_label = QLabel("Model Durumu: Eğitildi"); self.training_status_label.setStyleSheet("color: green; font-weight: bold;") # Bu satır GUI'de bir yere eklenmeli.
        # Eğitim bitince parametreleri tekrar aktif et
        self.set_params_enabled(True)
        self.save_button.setEnabled(True) # Model eğitildiği için kaydet butonu aktif olur.
        self.ai_button.setEnabled(True) # Model eğitildiği için AI ile oyna butonu aktif olur.
        
    def stop_training(self):
        # Eğitimi durdurur.
        if self.training_thread:
            self.training_thread.stop() # Eğitim thread'ine durma sinyali gönder.
            self.statusBar().showMessage("Eğitim durduruluyor...")
            self.set_params_enabled(True) # Parametreleri tekrar aktif et.
            # Butonları da uygun şekilde ayarlamak gerekebilir.
            self.train_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.set_game_buttons_enabled(True)

    def save_model(self):
        # Eğitilmiş DQN modelini kaydeder.
        save_dir = "models" # Kayıt dizini.
        if not os.path.exists(save_dir): os.makedirs(save_dir) # Dizin yoksa oluştur.
        # Dosya adı için zaman damgası ve grid boyutu kullanılır.
        timestamp = "dqn_model_" + str(self.grid_size) + "_" + str(random.randint(1000,9999)) + ".pth"
        filename, _ = QFileDialog.getSaveFileName(self, "DQN Modelini Kaydet", os.path.join(save_dir, timestamp), "PyTorch Files (*.pth);;All Files (*)") # Kayıt dialoğu.
        if filename: # Eğer bir dosya adı seçildiyse
            self.agent.save_model(filename) # Ajanın modelini kaydet.
            self.statusBar().showMessage(f"DQN modeli kaydedildi: {filename}")

    def load_model(self):
        # Kaydedilmiş bir DQN modelini yükler.
        filename, _ = QFileDialog.getOpenFileName(self, "DQN Modeli Yükle", "models" if os.path.exists("models") else ".", "PyTorch Files (*.pth);;All Files (*)") # Yükleme dialoğu.
        if filename: # Eğer bir dosya seçildiyse
            try:
                self.agent.load_model(filename) # Ajanın modelini yükle.
                self.model_loaded = True # Model yüklendi olarak işaretle.
                self.model_trained = True # Yüklenen model eğitilmiş sayılır.
                self.ai_button.setEnabled(True) # AI ile oyna butonunu aktif et.
                self.save_button.setEnabled(True) # Yüklenen model kaydedilebilir.
                self.statusBar().showMessage(f"DQN modeli başarıyla yüklendi: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Model Yükleme Hatası", f"Model yüklenirken bir hata oluştu: {e}")
                self.model_loaded = False
                self.model_trained = False
                self.ai_button.setEnabled(False)
                self.save_button.setEnabled(False)

    def play_with_ai(self):
        # Eğitilmiş veya yüklenmiş model ile AI'ın oynamasını başlatır.
        if self.game_timer.isActive(): # Eğer zamanlayıcı zaten aktifse (yani AI oynuyorsa)
            self.stop_game(); return # Oyunu durdur.
        if not self.model_trained and not self.model_loaded: # Eğer model yoksa
            QMessageBox.warning(self, "Model Yok", "AI ile oynamak için önce modeli eğitmeniz veya yüklemeniz gerekiyor.")
            return
        self.env.reset(); self.game_mode = 'ai'; self.info_panel.set_status("AI ile oynanıyor...")
        self.info_panel.clear_training_progress()  # AI ile oyna başlarken eğitim bilgisini gizle.
        self.ai_episode_count = 1; self.ai_total_reward = 0 # AI bölüm sayacını ve ödülünü sıfırla.
        self.game_timer.start(self.sim_speed) # Oyun zamanlayıcısını başlat.
        # Buton ve parametrelerin durumunu ayarla.
        self.stop_game_button.setEnabled(True)
        self.train_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.set_game_buttons_enabled(False) # Diğer oyun butonlarını pasif yap.
        self.ai_button.setEnabled(False) # AI ile oyna butonu zaten basıldığı için pasif.
        self.human_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.set_params_enabled(False) # Parametreleri pasif yap.

    def play_human_mode(self):
        # Kullanıcının manuel olarak oynamasını sağlar.
        if self.game_timer.isActive(): # Eğer AI oynuyorsa
            self.stop_game(); return # Oyunu durdur.
        self.env.reset(); self.game_mode = 'human'; # Ortamı sıfırla ve oyun modunu 'human' yap.
        self.info_panel.set_status("Manuel mod: Hareket=WASD/Ok Tuşları, Uç/Kalk/İn=Space, Kargo=E"); # Kullanıcıya bilgi ver.
        self.info_panel.clear_training_progress()  # İnsan modunda eğitim bilgisini gizle.
        self.update_ui() # Arayüzü güncelle.
        # Buton ve parametrelerin durumunu ayarla.
        self.stop_game_button.setEnabled(True)
        self.set_game_buttons_enabled(False) # Diğer oyun butonlarını pasif yap
        self.train_button.setEnabled(False)
        self.ai_button.setEnabled(False)
        self.human_button.setEnabled(False) # Manuel mod butonu zaten basıldığı için pasif.
        self.reset_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.set_params_enabled(False) # Parametreleri pasif yap.
        self.setFocus() # Klavye girdilerini almak için pencereye odaklan.

    def keyPressEvent(self, event):
        # Klavye tuşlarına basıldığında çağrılır (sadece manuel modda).
        if self.game_mode != 'human' or self.env.done: # Eğer manuel modda değilse veya bölüm bittiyse bir şey yapma.
            return
        key = event.key() # Basılan tuşu al.
        action = None # Başlangıçta eylem yok.
        # WASD ve ok tuşları ile hareket
        if key in (Qt.Key_Down, Qt.Key_S): # Aşağı
            action = 0
        elif key in (Qt.Key_Right, Qt.Key_D): # Sağa
            action = 1
        elif key in (Qt.Key_Up, Qt.Key_W): # Yukarı
            action = 2
        elif key in (Qt.Key_Left, Qt.Key_A): # Sola
            action = 3
        # Kargo al/bırak: E
        elif key == Qt.Key_E:
            action = 4
        # Kalk/İn: Space
        elif key == Qt.Key_Space:
            action = 5
        
        if action is not None: # Eğer geçerli bir eylem tuşuna basıldıysa
            _, _, done, info = self.env.step(action) # Eylemi uygula.
            self.update_ui() # Arayüzü güncelle.
            if "action" in info and info["action"]: # Eğer eylemle ilgili bir mesaj varsa durum çubuğunda göster.
                self.statusBar().showMessage(info["action"])
            if done: # Eğer bölüm bittiyse
                self.info_panel.set_status("Oyun bitti! Manuel modda yeni oyun için 'Sıfırla' veya 'Oyunu Durdur' kullanın.")
                # Oyun bittiğinde bazı butonları tekrar aktif hale getirebiliriz.
                self.stop_game_button.setEnabled(False) # Oyunu durdur butonu pasif.
                self.reset_button.setEnabled(True) # Sıfırla butonu aktif.
                self.human_button.setEnabled(True) # Tekrar manuel oynamak için.
                # Diğer butonlar da duruma göre ayarlanabilir.

    def stop_game(self):
        # AI veya manuel oyunu durdurur.
        if self.game_timer.isActive() or self.game_mode == 'human': # Eğer AI oynuyorsa veya manuel moddaysa
            self.game_timer.stop(); self.game_mode = None # Zamanlayıcıyı durdur ve oyun modunu sıfırla.
            self.info_panel.set_status("Oyun durduruldu.")
            self.info_panel.clear_training_progress()  # Oyun durunca eğitim bilgisini gizle.
            self.statusBar().showMessage("Oyun durduruldu.")
            # Buton ve parametrelerin durumunu eski haline getir.
            self.stop_game_button.setEnabled(False)
            self.stop_button.setEnabled(False) # Eğitim durdurma butonu da pasif olmalı.
            self.set_game_buttons_enabled(True) # Oyunla ilgili ana butonları aktif et.
            self.set_params_enabled(True) # Parametreleri aktif et.
# =====================
# Ana Uygulama Başlatıcı
# =====================
if __name__ == "__main__":
    # PyQt5 uygulamasını başlatır.
    app = QApplication(sys.argv)
    window = DroneDeliverySimulator() # Ana pencereyi oluştur.
    window.show() # Pencereyi göster.
    sys.exit(app.exec_()) # Uygulama döngüsünü başlat ve çıkışta temizle.