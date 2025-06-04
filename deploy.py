import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64

st.set_page_config(
    page_title="Cervic Cancer Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .cluster-result {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CervicalCancerDetector:
    def __init__(self):
        self.model = None
        self.features = []
        self.filenames = []
        self.labels = []
        
    def resize_image(self, img, width=224, height=224):
        """Resize image to specified dimensions"""
        resized = img.resize((width, height))
        return resized
    
    def rgb_to_hsv_manual(self, img):
        """Convert RGB to HSV and extract S channel"""
        img_np = np.asarray(img) / 255.0
        r, g, b = img_np[..., 0], img_np[..., 1], img_np[..., 2]
        
        cmax = np.max(img_np, axis=2)
        cmin = np.min(img_np, axis=2)
        delta = cmax - cmin
        
        s = np.where(cmax == 0, 0, delta / cmax)
        return s
    
    def mean_filter_3x3(self, img_array):
        """Apply 3x3 mean filter"""
        padded = np.pad(img_array, pad_width=1, mode='edge')
        result = np.zeros_like(img_array)
        
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                neighborhood = padded[i:i+3, j:j+3]
                result[i, j] = np.mean(neighborhood)
        
        return result
    
    def normalize_image(self, img_array):
        """Normalize image to 0-1 range"""
        norm = img_array - np.min(img_array)
        norm = norm / (np.max(norm) + 1e-6)
        return norm
    
    def extract_lbp_features(self, img_array, radius=1):
        """Extract LBP features"""
        n_points = 8 * radius
        lbp = local_binary_pattern(img_array, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist
    
    def extract_lbp_map(self, img_array, radius=1, method='uniform'):
        """Extract LBP map for visualization"""
        n_points = 8 * radius
        lbp = local_binary_pattern(img_array, n_points, radius, method=method)
        return lbp
    
    def process_image_manual(self, img):
        """Complete image processing pipeline"""
        # Resize image
        img_resized = self.resize_image(img)
        
        # Convert to HSV and extract S channel
        s_channel = self.rgb_to_hsv_manual(img_resized)
        
        # Apply mean filter
        s_blurred = self.mean_filter_3x3(s_channel)
        
        # Normalize
        s_normalized = self.normalize_image(s_blurred)
        
        # Extract LBP features
        lbp_feature = self.extract_lbp_features(s_normalized)
        
        return lbp_feature, {
            'original': img_resized,
            's_channel': s_channel,
            'blurred': s_blurred,
            'normalized': s_normalized,
            'lbp_map': self.extract_lbp_map(s_normalized)
        }
    
    def train_model(self, images_data):
        """Train K-means clustering model"""
        self.features = []
        self.filenames = []
        
        progress_container = st.container()
        progress_bar = progress_container.progress(0)
        status_text = progress_container.empty()
        
        for i, (filename, img) in enumerate(images_data):
            status_text.text(f'Processing image {i+1}/{len(images_data)}: {filename}')
            feature, _ = self.process_image_manual(img)
            self.features.append(feature)
            self.filenames.append(filename)
            progress_bar.progress((i + 1) / len(images_data))
        
        status_text.text('Training K-means model...')
        X = np.array(self.features)
        
        # Determine optimal number of clusters using elbow method
        sse = []
        K_range = range(1, min(11, len(X) + 1))
        
        for k in K_range:
            if k <= len(X):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X)
                sse.append(kmeans.inertia_)
        
        # For now, use 3 clusters (can be optimized)
        optimal_k = min(3, len(X))
        self.model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.labels = self.model.fit_predict(X)
        
        progress_bar.progress(1.0)
        status_text.text('✅ Model training completed!')
        
        return {
            'n_clusters': optimal_k,
            'silhouette_score': silhouette_score(X, self.labels) if len(set(self.labels)) > 1 else 0,
            'sse_values': sse,
            'k_range': list(K_range)
        }
    
    def predict(self, img):
        """Predict cluster for new image"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        feature, processing_steps = self.process_image_manual(img)
        cluster = self.model.predict([feature])[0]
        
        return cluster, processing_steps

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = CervicalCancerDetector()
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Main app
def main():
    st.title("Cervic Cancer Detection System")
    st.markdown("### Image Analysis using Local Binary Pattern & K-Means Clustering")
    
    # Sidebar
    st.sidebar.header("Navigation")
    mode = st.sidebar.selectbox(
        "Select Mode:",
        ["Home", "Train Model", "Predict Image", "Visualizations"]
    )
    
    if mode == "Home":
        show_home()
    elif mode == "Train Model":
        show_training()
    elif mode == "Predict Image":
        show_prediction()
    elif mode == "Visualizations":
        show_visualizations()

def show_home():
    st.header("Welcome to Cervical Cancer Detection System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("About This System")
        st.write("""
        This system uses advanced computer vision techniques to analyze cervical cancer images:
        
        **Key Features:**
        - **Local Binary Pattern (LBP)** for texture analysis
        - **K-Means Clustering** for pattern grouping
        - **HSV Color Space** processing
        - **Interactive visualizations**
        
        **Process Flow:**
        1. Image preprocessing (resize, HSV conversion)
        2. Feature extraction using LBP
        3. K-means clustering for pattern recognition
        4. Cluster prediction for new images
        """)
    
    with col2:
        st.subheader("Getting Started")
        st.write("""
        **Step 1: Train Model**
        - Upload multiple cervical cancer images
        - System will extract LBP features
        - K-means clustering will group similar patterns
        
        **Step 2: Predict New Images**
        - Upload a single image for analysis
        - Get cluster assignment and processing visualization
        
        **Step 3: View Results**
        - Analyze clustering results
        - View elbow plots and PCA visualizations
        """)
        
        if st.session_state.model_trained:
            st.success("✅ Model is trained and ready for predictions!")
        else:
            st.warning("Model not trained yet. Please train the model first.")

def show_training():
    st.header("Train Clustering Model")
    
    st.write("Upload multiple images to train the K-means clustering model.")
    
    uploaded_files = st.file_uploader(
        "Choose training images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload multiple cervical cancer images for training"
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} files for training")
        
        # Preview images
        if st.checkbox("Preview uploaded images"):
            cols = st.columns(3)
            for i, file in enumerate(uploaded_files[:6]):  # Show max 6 images
                with cols[i % 3]:
                    img = Image.open(file).convert('RGB')
                    st.image(img, caption=file.name, use_column_width=True)
        
        if st.button("Start Training", type="primary"):
            try:
                # Prepare training data
                images_data = []
                for file in uploaded_files:
                    img = Image.open(file).convert('RGB')
                    images_data.append((file.name, img))
                
                # Train model
                with st.spinner("Training model... This may take a few minutes."):
                    results = st.session_state.detector.train_model(images_data)
                
                st.session_state.model_trained = True
                
                # Display results
                st.success("Model training completed successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Images Processed", len(uploaded_files))
                with col2:
                    st.metric("Number of Clusters", results['n_clusters'])
                with col3:
                    st.metric("Silhouette Score", f"{results['silhouette_score']:.3f}")
                
                # Show cluster assignments
                st.subheader("Cluster Assignments")
                cluster_data = []
                for i, filename in enumerate(st.session_state.detector.filenames):
                    cluster_data.append({
                        'Filename': filename,
                        'Cluster': f"Cluster {st.session_state.detector.labels[i] + 1}"
                    })
                
                df = pd.DataFrame(cluster_data)
                st.dataframe(df, use_container_width=True)
                
                # Elbow plot
                if len(results['sse_values']) > 1:
                    st.subheader("Elbow Method Analysis")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(results['k_range'], results['sse_values'], 'bx-')
                    ax.set_xlabel('Number of Clusters (k)')
                    ax.set_ylabel('Inertia / SSE')
                    ax.set_title('Elbow Method for Optimal k')
                    ax.grid(True)
                    st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")

def show_prediction():
    st.header("Predict New Image")
    
    if not st.session_state.model_trained:
        st.warning("Please train the model first before making predictions.")
        return
    
    st.write("Upload a single image to get cluster prediction and processing visualization.")
    
    uploaded_file = st.file_uploader(
        "Choose an image for prediction",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a cervical cancer image for cluster prediction"
    )
    
    if uploaded_file:
        # Display uploaded image
        img = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Predict Cluster", type="primary"):
            try:
                with st.spinner("Processing image and making prediction..."):
                    cluster, processing_steps = st.session_state.detector.predict(img)
                
                # Display prediction result
                with col2:
                    st.markdown(f"""
                    <div class="cluster-result">
                         Prediction Result: Cluster {cluster + 1}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show processing steps
                st.subheader("Image Processing Steps")
                
                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                axes = axes.ravel()
                
                # Original image
                axes[0].imshow(processing_steps['original'])
                axes[0].set_title('1. Original Image (224x224)')
                axes[0].axis('off')
                
                # S channel
                axes[1].imshow(processing_steps['s_channel'], cmap='gray')
                axes[1].set_title('2. S Channel from HSV')
                axes[1].axis('off')
                
                # Blurred
                axes[2].imshow(processing_steps['blurred'], cmap='gray')
                axes[2].set_title('3. Mean Filter 3x3')
                axes[2].axis('off')
                
                # Normalized
                axes[3].imshow(processing_steps['normalized'], cmap='gray')
                axes[3].set_title('4. Normalized')
                axes[3].axis('off')
                
                # LBP map
                axes[4].imshow(processing_steps['lbp_map'], cmap='gray')
                axes[4].set_title('5. LBP Map')
                axes[4].axis('off')
                
                # Remove empty subplot
                axes[5].remove()
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")

def show_visualizations():
    st.header("Model Visualizations")
    
    if not st.session_state.model_trained:
        st.warning("Please train the model first to view visualizations.")
        return
    
    detector = st.session_state.detector
    X = np.array(detector.features)
    
    tab1, tab2, tab3 = st.tabs(["PCA Visualization", "t-SNE Visualization", "Feature Analysis"])
    
    with tab1:
        st.subheader("Principal Component Analysis (PCA)")
        
        if len(X) > 1:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = ['red', 'green', 'blue', 'orange', 'purple']
            
            for i in range(len(set(detector.labels))):
                mask = detector.labels == i
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          c=colors[i % len(colors)], label=f'Cluster {i+1}', alpha=0.7, s=50)
            
            ax.set_title("K-means Clustering Visualization (PCA)", fontsize=14)
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # PCA explanation
            st.write(f"**Explained Variance Ratio:** {pca.explained_variance_ratio_}")
            st.write(f"**Total Explained Variance:** {sum(pca.explained_variance_ratio_):.3f}")
        else:
            st.info("Need more than 1 sample for PCA visualization")
    
    with tab2:
        st.subheader("t-SNE Visualization")
        
        if len(X) > 1:
            with st.spinner("Computing t-SNE... This may take a moment."):
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
                X_tsne = tsne.fit_transform(X)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = ['red', 'green', 'blue', 'orange', 'purple']
            
            for i in range(len(set(detector.labels))):
                mask = detector.labels == i
                ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                          c=colors[i % len(colors)], label=f'Cluster {i+1}', alpha=0.7, s=50)
            
            # Add annotations
            for i, filename in enumerate(detector.filenames):
                ax.annotate(filename, (X_tsne[i, 0], X_tsne[i, 1]), 
                           fontsize=8, alpha=0.7)
            
            ax.set_title("t-SNE Visualization of LBP Features", fontsize=14)
            ax.set_xlabel("t-SNE Dimension 1")
            ax.set_ylabel("t-SNE Dimension 2")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("Need more than 1 sample for t-SNE visualization")
    
    with tab3:
        st.subheader("Feature Analysis")
        
        # Cluster statistics
        cluster_stats = []
        for i in range(len(set(detector.labels))):
            mask = detector.labels == i
            cluster_features = X[mask]
            cluster_stats.append({
                'Cluster': f'Cluster {i+1}',
                'Count': int(np.sum(mask)),
                'Mean Feature': float(np.mean(cluster_features)),
                'Std Feature': float(np.std(cluster_features))
            })
        
        df_stats = pd.DataFrame(cluster_stats)
        st.dataframe(df_stats, use_container_width=True)
        
        # Feature distribution
        st.subheader("LBP Feature Distribution by Cluster")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i in range(len(set(detector.labels))):
            mask = detector.labels == i
            cluster_features = X[mask]
            ax.hist(cluster_features.flatten(), alpha=0.5, 
                   label=f'Cluster {i+1}', bins=30)
        
        ax.set_xlabel('LBP Feature Values')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of LBP Features by Cluster')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

if __name__ == "__main__":
    main()