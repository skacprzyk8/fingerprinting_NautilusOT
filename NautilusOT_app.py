'''Section with all necessary imports'''
# GUI and File Dialogs
import tkinter as tk
from tkinter import filedialog as fd
# Data Processing
import pandas as pd
import os
import re
from ipaddress import ip_network
# ML and Encoding
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# Plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class MultiCSVApp:
    """
    A GUI application that allows users to:
    - Load and preprocess multiple CSV files from specific clients network
    - Train a RandomForestClassifier
    - Predict categories for unlabelled data
    - Display and save results

    Notes:
    - The asset dataset must contain a column named 'TRAINING'.
      - A value of 1 indicates the row will be used for training the model.
      - A value of 0 indicates the row will be classified using the trained model.
    """
    # Initialize main window

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Multi CSV Loader")
        self.window.geometry("800x600")
        # Internal storage for loaded DataFrames
        self.data_frames = {}
        # CSV-assigned DataFrames
        self.asset = None
        self.prop = None
        self.relservicemetric = None
        self.relservices = None
        self.relations = None
        self.types = None
        self.df = None
        self.data = None
        self.label_encoder = None
        self.canvas = None
        # GUI widgets
        self.upload_button = tk.Button(
            self.window, text="Upload CSV Files", command=self.load_multiple_csvs)
        self.upload_button.pack(pady=10)
        self.train_button = tk.Button(
            self.window, text="Train & Classify", command=self.train_and_classify, state=tk.DISABLED)
        self.train_button.pack(pady=10)
        # Text area to show output and logs
        text_frame = tk.Frame(self.window)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.result = tk.Text(text_frame, height=15,
                              width=100, yscrollcommand=scrollbar.set)
        self.result.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.result.yview)

        self.window.mainloop()

    def network_to_int(self, ip):
        """
        Convert IP address string to an integer. 
        """
        try:
            return int(ip_network(ip, strict=False).network_address)
        except:
            return None

    def mac_to_int(self, mac):
        """
        Convert MAC address to an integer by removing delimiters.
        """
        try:
            return int(mac.replace(":", "").replace("-", "").lower(), 16)
        except:
            return None

    def extract_first_vlan(self, vlan_str):
        """
        Extract the first VLAN ID from a string of comma-separated VLANs.
        """
        try:
            if pd.isna(vlan_str) or vlan_str.strip('{}').strip() == '':
                return 0
            return int(vlan_str.strip('{}').split(',')[0].strip())
        except:
            return None

    def load_multiple_csvs(self):
        """
            Loads and categorizes multiple CSV files selected by the user.
            Assigns each file to a corresponding internal DataFrame variable based on filename suffix patterns:

            Expected filename patterns (where * is a wildcard for clients network name):
            - '*_assets'                          → assigned to self.asset
            - '*_asset_properties'               → assigned to self.prop
            - '*_asset_relationship_service_metrics' → assigned to self.relservicemetric
            - '*_asset_relationship_services'    → assigned to self.relservices
            - '*_asset_relationships'            → assigned to self.relations
            - '*_asset_types'                    → assigned to self.types

            Files that do not match these patterns will still be loaded but not categorized.
        """
        file_paths = fd.askopenfilenames(
            title="Select CSV files", filetypes=[("CSV Files", "*.csv")])
        for path in file_paths:
            try:
                df = pd.read_csv(path)
                file_name = os.path.splitext(os.path.basename(path))[0]

                self.data_frames[file_name] = df
                if re.search(r'_assets$', file_name):
                    self.asset = df
                    self.result.insert(
                        tk.END, f"Assigned to: asset ({file_name})\n")
                elif re.search(r'_asset_properties$', file_name):
                    self.prop = df
                    self.result.insert(
                        tk.END, f"Assigned to: prop ({file_name})\n")
                elif re.search(r'_asset_relationship_service_metrics$', file_name):
                    self.relservicemetric = df
                    self.result.insert(
                        tk.END, f"Assigned to: relservicemetric ({file_name})\n")
                elif re.search(r'_asset_relationship_services$', file_name):
                    self.relservices = df
                    self.result.insert(
                        tk.END, f"Assigned to: relservices ({file_name})\n")
                elif re.search(r'_asset_relationships$', file_name):
                    self.relations = df
                    self.result.insert(
                        tk.END, f"Assigned to: relations ({file_name})\n")
                elif re.search(r'_asset_types$', file_name):
                    self.types = df
                    self.result.insert(
                        tk.END, f"Assigned to: types ({file_name})\n")
                else:
                    self.result.insert(
                        tk.END, f"Loaded (unmatched): {file_name} ({df.shape[0]} rows, {df.shape[1]} cols)\n")
            except Exception as e:
                self.result.insert(tk.END, f"Failed to load {path}: {e}\n")
         # Run preprocessing after loading files
        self.preprocess_data()
        self.train_button.config(state=tk.NORMAL)

    def preprocess_data(self):
        """
        Prepares and merges data from the CSVs.
        Cleans fields, encodes categories, and builds features.
        """
        try:
            asset = self.asset
            prop = self.prop
            relations = self.relations
            types = self.types
            relservicemetric = self.relservicemetric.copy()
            relservices = self.relservices.copy()
            # Drop unused columns and clean asset data
            asset = asset.drop(columns=[
                               'name', 'hostname', 'active', 'location_id', 'assigned_value', 'estimated_value'])
            asset = asset.dropna(subset=['asset_type_id'])
            # Map processing_state to binary
            asset['processing_state_binary'] = asset['processing_state'].map({
                'QUEUED': 1,
                'PROCESSED': 0
            })
            asset = asset.drop(columns=['processing_state'])
            # Convert IP/MAC/VLAN features to integers
            asset['ipaddress'] = asset['ipaddress'].apply(self.network_to_int)
            asset['mac_address'] = asset['mac_address'].apply(self.mac_to_int)
            asset['vlan_ids'] = asset['vlan_ids'].apply(
                self.extract_first_vlan)
            # Calculate device age
            asset['date_created'] = pd.to_datetime(
                asset['date_created'], utc=True).dt.tz_convert('UTC')
            asset['date_updated'] = pd.to_datetime(
                asset['date_updated'], utc=True).dt.tz_convert('UTC')
            asset['device_age_days'] = (
                asset['date_updated'] - asset['date_created']).dt.days
            asset = asset.drop(
                columns=['date_created', 'date_updated', 'last_reported'])
            types = types.dropna(subset='category')
            types = types.drop(columns=['name', 'description', 'active', 'parent_asset_type_id',
                               'match_slug', 'icon', 'date_created', 'date_updated'])
            # Clean and encode properties
            prop['key'] = prop['key'].str.replace('^snmp_', '', regex=True)
            prop['key'] = prop['key'].str.split('.').apply(
                lambda x: [int(i) for i in x if i != ''])
            prop['key'] = prop['key'].apply(
                lambda x: int(''.join(map(str, x))))
            prop["value"] = prop["value"].map(prop["value"].value_counts())
            prop = prop.drop(
                columns=['date_created', 'date_updated', 'display_name'])
            relations = relations.drop(
                columns=['id', 'date_created', 'date_updated', 'tags'])
            relservices = relservices.drop(
                columns=['date_created', 'date_updated'])
            relservice_metrics = relservicemetric.drop(
                columns=['key', 'attributes', 'date_created', 'date_updated'])
            # MERGING
            label_encoder = LabelEncoder()
            df = asset.merge(types, how='left', left_on='asset_type_id',
                             right_on='id', suffixes=('', '_type'))
            df = df.drop(columns=['id_type'])
            df = df.merge(prop, how='left', left_on='id',
                          right_on='asset_id', suffixes=('', '_prop'))
            df['value'] = df['value'].fillna(0)
            # Source and target counts
            source_counts = relations['source_asset_id'].value_counts().rename(
                'source_count')
            target_counts = relations['target_asset_id'].value_counts().rename(
                'target_count')
            df['source_count'] = df['id'].map(
                source_counts).fillna(0).astype(int)
            df['target_count'] = df['id'].map(
                target_counts).fillna(0).astype(int)
            df = df.drop(columns=['asset_id', 'id_prop'])
            # Parent-child counts
            child_counts = df['parent_asset_id'].value_counts().rename(
                'child_count')
            df['child_count'] = df['id'].map(
                child_counts).fillna(0).astype(int)
            df['has_parent'] = df['parent_asset_id'].notna().astype(int)
            # Encode vendor
            vendor_counts = df['vendor'].value_counts()
            df['vendor'] = df['vendor'].map(vendor_counts)
            df['vendor'] = label_encoder.fit_transform(
                df['vendor'].astype(str))
            data = df.drop(columns=['parent_asset_id',
                           'id', 'asset_type_id']).fillna(0)
            data['category_encoded'] = label_encoder.fit_transform(
                df['category'])
            data = data.drop(columns=['category'])
            self.data = data
            self.label_encoder = label_encoder
        except Exception as e:
            self.result.insert(tk.END, f"Preprocessing error: {e}\n")

    def train_and_classify(self):
        """
        Trains a RandomForestClassifier using the preprocessed data.
        Predicts and displays the result distribution.
        """
        try:
            if self.data is None or self.label_encoder is None:
                self.result.insert(
                    tk.END, "No preprocessed data available. Please upload and preprocess CSVs first.\n")
                return
            data = self.data.copy()

            X = data.drop(columns=['category_encoded'])
            y = data['category_encoded']

            if 'TRAINING' not in data.columns:
                raise ValueError("Missing 'TRAINING' column.")
            # Split into training and required classification sets
            X_train = X[X['TRAINING'] == 1].drop(columns='TRAINING')
            y_train = y[X['TRAINING'] == 1]

            X_test = X[X['TRAINING'] == 0].drop(columns='TRAINING')
            y_test = y[X['TRAINING'] == 0]
            # Train classifier
            rf = RandomForestClassifier(random_state=42)
            rf.fit(X_train, y_train)

            self.data['result'] = self.label_encoder.inverse_transform(
                self.data['category_encoded'])

            non_training_indices = data[data['TRAINING'] == 0].index
            # Predict categories
            predicted_categories_encoded = rf.predict(X_test)
            predicted_categories = self.label_encoder.inverse_transform(
                predicted_categories_encoded)
            # Add predictions to data
            data.loc[non_training_indices, 'result'] = predicted_categories
            preds_encoded = rf.predict(X_test)
            preds = self.label_encoder.inverse_transform(preds_encoded)
            data.loc[data['TRAINING'] == 0, 'result'] = preds

            # Show prediction distribution
            value_counts = pd.Series(predicted_categories).value_counts()
            self.result.insert(
                tk.END, "\nPredicted Category Distribution (value_counts):\n")
            self.result.insert(tk.END, f"{value_counts.to_string()}\n")
            save_path = fd.asksaveasfilename(defaultextension=".csv", filetypes=[
                                             ("CSV files", "*.csv")], title="Save Result CSV")
            # Remove previous chart if it exists
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
            # Plot results
            fig, ax = plt.subplots(figsize=(6, 4))
            value_counts.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title('Predicted Category Distribution')
            ax.set_xlabel('Category')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)

            self.canvas = FigureCanvasTkAgg(fig, master=self.window)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(pady=10)
            if save_path:
                data.to_csv(save_path, index=False)
                self.result.insert(tk.END, f"Results saved to: {save_path}\n")
        except Exception as e:
            self.result.insert(
                tk.END, f"Error during training/classification: {e}\n")


# Launch the app
app = MultiCSVApp()
app.window.mainloop()
