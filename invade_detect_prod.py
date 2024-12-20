import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model

class AttackDetector:
    def __init__(self, model_path, scaler_path):
        """Inicjalizacja modelu, skalera oraz buforów."""
        # Wczytaj model i skaler
        self.model = load_model(model_path)
        
        # Wczytaj skaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Wczytaj enkodery
        self.encoders = {}
        for col in ['protocol_type', 'service', 'flag']:
            with open(f'{col}_encoder.pkl', 'rb') as f:
                self.encoders[col] = pickle.load(f)
        
        # Bufor do śledzenia dodatkowych cech
        self.host_activity = {}  # Słownik do śledzenia aktywności hostów

    def track_host_service(self, src_ip, dst_ip, service):
        """Śledzenie liczby połączeń z adresu IP źródłowego do adresu IP docelowego na określony serwis."""
        key = (src_ip, dst_ip, service)
        
        # Jeśli para (src_ip, dst_ip, service) nie istnieje, zainicjalizuj ją
        if key not in self.host_activity:
            self.host_activity[key] = 0
        
        # Zwiększ licznik połączeń dla danej pary (src_ip, dst_ip, service)
        self.host_activity[key] += 1

    def get_monitored_features(self, dst_ip, service):
        count = sum(count for (src_ip, dst, service), count in self.host_activity.items() if dst == dst_ip)
        srv_count = sum(count for (src_ip, dst, service_) , count in self.host_activity.items() if service_ == service)
        total_connections = sum(count for (src_ip, dst, service_), count in self.host_activity.items())
        same_srv_rate = srv_count / total_connections
        diff_srv_rate = 1 - same_srv_rate

        dst_host_count = sum(count for (src_ip, dst, service_), count in self.host_activity.items() if dst == dst_ip)
        dst_host_srv_count = sum(count for (src_ip, dst, service_), count in self.host_activity.items() if dst == dst_ip and service_ == service)
        dst_host_same_srv_rate = dst_host_srv_count / dst_host_count if dst_host_count > 0 else 0
        dst_host_diff_srv_rate = 1 - dst_host_same_srv_rate
        
        srv_diff_host_count = sum(count for (src_ip, dst, service_), count in self.host_activity.items() if service_ == service and dst != dst_ip)
        srv_diff_host_rate = srv_diff_host_count / srv_count if srv_count > 0 else 0
        
        return count, srv_count, same_srv_rate, diff_srv_rate, dst_host_count, dst_host_srv_count, dst_host_diff_srv_rate, srv_diff_host_rate
  
    def prepare_packet_features(self, packet):
        """Przygotowanie cech pakietu do predykcji."""
        # Kodowanie cech kategorycznych
        for col in ['protocol_type', 'service', 'flag']:
            try:
                packet[col] = self.encoders[col].transform([packet[col]])[0]
            except ValueError:
                # Jeśli etykieta nieznana, użyj ostatniej znanej kategorii
                packet[col] = len(self.encoders[col].classes_) - 1
        
        # Śledzenie aktywności hosta
        src_ip = packet.get('src_ip', '0.0.0.0')
        dst_ip = packet.get('dst_ip', '0.0.0.0')

        # Zapisuje liczbę połączeń z danego IP do serwisu
        self.track_host_service(packet['src_ip'], packet['service'])  
        
        land = 1 if src_ip == dst_ip else 0
        
        # Lista cech w dokładnie takiej kolejności jak podczas treningu
        features = [
            packet.get('duration', 0),
            packet['protocol_type'],
            packet['service'], 
            packet['flag'],
            packet.get('src_bytes', 0),
            packet.get('dst_bytes', 0),
            land,  # cecha 'land'
            packet.get('wrong_fragment', 0),
            packet.get('urgent', 0),
            packet.get('hot', 0),
            1 if packet.get('logged_in', False) else 0,
            # num_compromised,  # cecha 'num_compromised'
            # count,  # cecha 'count'
            packet.get('srv_count', 0),
            packet.get('serror_rate', 0),
            packet.get('rerror_rate', 0),
            packet.get('same_srv_rate', 0),
            packet.get('diff_srv_rate', 0),
            packet.get('srv_diff_host_rate', 0),
            packet.get('dst_host_count', 0),
            packet.get('dst_host_srv_count', 0),
            packet.get('dst_host_same_srv_rate', 0),
            packet.get('dst_host_diff_srv_rate', 0)
        ]
        
        # Konwersja na numpy array
        features = np.array(features).reshape(1, -1)
        
        # Normalizacja cech
        features = self.scaler.transform(features)
        
        # Reshape dla LSTM
        features = features.reshape((1, 1, features.shape[1]))
        
        return features
    
    def detect_attack(self, packet):
        """Wykrywanie ataku dla pojedynczego pakietu."""
        # Przygotowanie cech pakietu
        features = self.prepare_packet_features(packet)
        
        # Predykcja
        prediction = self.model.predict(features, verbose=0)
        
        # Konwersja predykcji
        is_attack = np.argmax(prediction) == 1
        
        # Ustawienie flagi 'is_attack' dla pakietu
        packet['is_attack'] = is_attack
        
        return is_attack, prediction[0]

# Przykładowe użycie
if __name__ == '__main__':
    detector = AttackDetector('lstm_model.h5', 'scaler.pkl')
    
    # Przykładowy pakiet przychodzący
    incoming_packet = {
        'duration': 2,
        'protocol_type': 'TCP',
        'service': 'HTTP',
        'flag': 'SYN',
        'src_bytes': 1024,
        'dst_bytes': 512,
        'wrong_fragment': 0,
        'urgent': 0,
        'logged_in': False,
        'hot': 0,
        'src_ip': '192.168.1.10',
        'dst_ip': '192.168.1.20'
    }
    
    is_attack, prediction = detector.detect_attack(incoming_packet)
    print(f"Atak wykryty: {is_attack}")
    print(f"Predykcja: Normalny: {prediction[0]:.4f}, Atak: {prediction[1]:.4f}")

    syn_flood_packet = {
        'duration': 0.1,  # Krótkotrwałe połączenie
        'protocol_type': 'TCP',
        'service': 'HTTP',
        'flag': 'SYN',  # Sama flaga SYN sugeruje możliwy atak
        'src_bytes': 0,  # Minimalna ilość danych
        'dst_bytes': 0,
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 10,  # Wysoka liczba pilnych pakietów
        'hot': 1,  # Wskaźnik potencjalnego ataku
        'logged_in': False,
        'num_compromised': 5,  # Wysoka liczba zagrożonych warunków
        'count': 50,  # Duża liczba połączeń
        'srv_count': 30,  # Wiele połączeń do tej samej usługi
        'serror_rate': 0.8,  # Wysoki współczynnik błędów SYN
        'rerror_rate': 0.6,
        'same_srv_rate': 0.9,
        'diff_srv_rate': 0.1,
        'srv_diff_host_rate': 0.2,
        'dst_host_count': 1,
        'dst_host_srv_count': 1,
        'dst_host_same_srv_rate': 0.9,
        'dst_host_diff_srv_rate': 0.1
    }

    port_scan_packet = {
        'duration': 5,  # Dłuższy czas trwania
        'protocol_type': 'TCP',
        'service': 'Unknown',  # Nieznana usługa
        'flag': 'REJ',  # Odrzucone połączenie
        'src_bytes': 100,
        'dst_bytes': 50,
        'land': 0,
        'wrong_fragment': 3,  # Kilka nieprawidłowych fragmentów
        'urgent': 0,
        'hot': 1,
        'logged_in': False,
        'num_compromised': 3,
        'count': 40,  # Wiele połączeń
        'srv_count': 10,
        'serror_rate': 0.5,
        'rerror_rate': 0.7,
        'same_srv_rate': 0.2,
        'diff_srv_rate': 0.8,  # Wysoki współczynnik różnych usług
        'srv_diff_host_rate': 0.6,
        'dst_host_count': 10,  # Połączenia do wielu hostów
        'dst_host_srv_count': 2,
        'dst_host_same_srv_rate': 0.1,
        'dst_host_diff_srv_rate': 0.9
    }

    data_exfil_packet = {
        'duration': 300,  # Bardzo długie połączenie
        'protocol_type': 'TCP',
        'service': 'FTP',
        'flag': 'ACC',  # Zaakceptowane połączenie
        'src_bytes': 10000,  # Duża ilość wysłanych danych
        'dst_bytes': 500,
        'land': 0,
        'wrong_fragment': 0,
        'urgent': 0,
        'hot': 1,
        'logged_in': True,  # Zalogowany użytkownik
        'num_compromised': 2,
        'count': 5,
        'srv_count': 3,
        'serror_rate': 0.1,
        'rerror_rate': 0.1,
        'same_srv_rate': 0.8,
        'diff_srv_rate': 0.2,
        'srv_diff_host_rate': 0.1,
        'dst_host_count': 1,
        'dst_host_srv_count': 1,
        'dst_host_same_srv_rate': 0.9,
        'dst_host_diff_srv_rate': 0.1
    }

    test_packets = [syn_flood_packet, port_scan_packet, data_exfil_packet]

    for i, packet in enumerate(test_packets, 1):
        is_attack, prediction = detector.detect_attack(packet)
        print(f"Pakiet {i}:")
        print(f"Atak wykryty: {is_attack}")
        print(f"Prawdopodobieństwo ataku: {prediction[1]:.4f}\n")