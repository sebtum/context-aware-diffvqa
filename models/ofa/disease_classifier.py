import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

# Helper function for linear layers with Xavier uniform initialization.
def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class DiseaseClassifier(nn.Module):
    """
    Disease classifier module that takes ResNet101 output features of shape [B, 1024, 24, 24]
    and produces multi-label logits for multiple diseases.
    
    It supports two pooling methods:
      - 'attn': Attention-based pooling over patches.
      - 'avg': Adaptive average pooling.
      
    It also supports two classifier options:
      - 'linear': A single linear projection.
      - 'mlp': A two-layer MLP with a non-linear activation (e.g. tanh).
      
    Multi-label classification implies that the final activation is left as logits 
    (to be used with BCEWithLogitsLoss during training).
    """
    def __init__(self, num_diseases, in_channels=1024, pooling='attn', 
                 classifier_type='linear', dropout_prob=0.5, attn_hidden_dim=128, 
                 mlp_hidden_dim=256, activation='tanh', use_spectral_norm=False):
        super(DiseaseClassifier, self).__init__()
        self.pooling = pooling
        self.classifier_type = classifier_type
        self.in_channels = in_channels

        # If using attention pooling, define the attention mechanism.
        if self.pooling == 'attn':
            # We'll learn an attention score for each patch.
            # First layer: project each patch feature to a hidden dimension.
            self.attn_fc = Linear(in_channels, attn_hidden_dim)
            # Second layer: output a single score per patch.
            self.attn_score = Linear(attn_hidden_dim, 1)
        elif self.pooling == 'avg':
            # Adaptive average pooling over spatial dimensions.
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError("Unknown pooling type: choose 'attn' or 'avg'")
        
        self.dropout = nn.Dropout(dropout_prob)
        
        # Define classifier head: either a simple linear or an MLP.
        if self.classifier_type == 'linear':
            fc = nn.Linear(in_channels, num_diseases, bias=True)
            if use_spectral_norm:
                fc = nn.utils.spectral_norm(fc)
            self.classifier = fc
        elif self.classifier_type == 'mlp':
            layers = []
            layers.append(nn.Linear(in_channels, mlp_hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            else:
                raise ValueError("Unsupported activation. Choose 'tanh' or 'relu'")
            layers.append(nn.Dropout(dropout_prob))
            fc = nn.Linear(mlp_hidden_dim, num_diseases, bias=True)
            if use_spectral_norm:
                fc = nn.utils.spectral_norm(fc)
            layers.append(fc)
            self.classifier = nn.Sequential(*layers)
        else:
            raise ValueError("Unknown classifier type: choose 'linear' or 'mlp'")
        
    def forward(self, x):
        """
        Args:
            x: ResNet101 output features with shape [B, N, C]
               For example, [B, 24x24, 1024].
        Returns:
            logits: A tensor of shape [B, num_diseases], where each entry is the logit
                    for the corresponding disease (to be used with multi-label loss).
        """
        B, N, C = x.size()  # For example, C=1024, H=W=24.
        if self.pooling == 'attn':
            # Compute attention scores:
            # Pass each patch feature through an MLP.
            attn_hidden = torch.tanh(self.attn_fc(x))  # [B, N, attn_hidden_dim]
            attn_logits = self.attn_score(attn_hidden)       # [B, N, 1]
            attn_weights = F.softmax(attn_logits, dim=1)      # [B, N, 1]
            # Weighted sum of patch features:
            pooled = (x * attn_weights).sum(dim=1)       # [B, C]
        elif self.pooling == 'avg':
            # Use adaptive average pooling to collapse spatial dimensions.
            pooled = self.avgpool(x)       # [B, C, 1, 1]
            pooled = pooled.view(B, C)     # [B, C]
        else:
            raise ValueError("Unknown pooling type")
        
        pooled = self.dropout(pooled)
        
        # Classify using the chosen head.
        logits = self.classifier(pooled)  # [B, num_diseases]
        return logits


class LocationClassifier(nn.Module):
    def __init__(self, img_dim, disease_emb_dim, attn_dim, hidden_dim, location_dim, num_diseases):
        """
        Args:
            img_dim (int): Dimensionality of each image patch feature.
            disease_emb_dim (int): Dimensionality of the disease embedding.
            attn_dim (int): Dimensionality for the cross-attention projection.
            hidden_dim (int): Hidden layer size for fusion.
            location_dim (int): Number of location logits per disease.
            num_diseases (int): Total number of diseases.
        """
        super(LocationClassifier, self).__init__()
        # Projection layers for cross-attention.
        self.fc_key = Linear(img_dim, attn_dim)
        self.fc_value = Linear(img_dim, attn_dim)
        self.fc_query = Linear(disease_emb_dim, attn_dim)
        
        # Disease embeddings for all diseases.
        self.disease_embed = nn.Embedding(num_diseases, disease_emb_dim)
        # Learnable no-disease embedding.
        self.no_disease_embedding = nn.Parameter(torch.empty(disease_emb_dim))
        nn.init.xavier_uniform_(self.no_disease_embedding.unsqueeze(0))
        self.num_diseases = num_diseases

        # Layer normalization for the fused features.
        self.ln_fused = nn.LayerNorm(attn_dim + disease_emb_dim)
        
        # Fusion and final classifier.
        self.classifier = nn.Sequential(
            nn.Linear(attn_dim + disease_emb_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, location_dim)
        )
    
    def forward(self, image_patch_features, disease_logits=None, gold_disease_labels=None, threshold=0.5):
        """
        Args:
            image_patch_features: Tensor of shape [B, N, img_dim] (N image patches per sample).
            disease_logits: Optional tensor of shape [B, num_diseases] from the disease classifier.
            gold_disease_labels: Optional tensor of shape [B] with ground-truth disease indices.
        Returns:
            logits: Tensor of shape [B, num_diseases, location_dim] representing location logits per disease.
        """
        B = image_patch_features.size(0)
        N = image_patch_features.size(1)
        
        # Expand the disease embeddings to each sample.
        # Original disease embeddings: [num_diseases, disease_emb_dim]
        # Expanded: [B, num_diseases, disease_emb_dim]
        disease_emb = self.disease_embed.weight.unsqueeze(0).expand(B, -1, -1)
        # Expand the no-disease embedding to shape [B, num_diseases, disease_emb_dim]
        no_disease_emb = self.no_disease_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, disease_emb_dim]
        no_disease_emb = no_disease_emb.expand(B, self.num_diseases, -1)

        # Create a filtering mask per sample:
        # If gold labels are provided, create a one-hot mask.
        # Else if disease logits are provided, convert them to probabilities.
        if gold_disease_labels is not None:
            mask = gold_disease_labels
            # Expand mask to match the embedding dimension: [B, num_diseases, 1]
            mask = mask.unsqueeze(-1)
            final_disease_emb = disease_emb * mask + (1 - mask) * no_disease_emb
        elif disease_logits is not None:
            # We'll combine the hard decision with the soft probabilities:
            # For diseases present (hard_mask == 1), we use the product of the disease embedding and the soft probability.
            # For diseases absent (hard_mask == 0), we use a dedicated "no disease" embedding.
            # Get the soft probability as well.
            prob = torch.sigmoid(disease_logits).unsqueeze(-1)  # [B, num_diseases, 1]
            # Derive a hard binary mask from the soft probabilities.
            hard_mask = (prob > threshold).float()  # [B, num_diseases, 1]
            
            # For each disease, if the hard decision indicates presence (hard_mask==1), use the disease embedding scaled by soft prob.
            # Otherwise, use the no-disease embedding.
            final_disease_emb = hard_mask * (disease_emb * prob) + (1 - hard_mask) * no_disease_emb
        else:
            # If no filtering info is provided, use a uniform mask.
            warnings.warn("LocationClassifier: No filtering info provided; falling back to a uniform mask.", UserWarning)
            mask = torch.ones(B, self.num_diseases, device=image_patch_features.device)
            final_disease_emb = disease_emb * mask
        
        # Project the filtered disease embeddings to query space.
        # queries: [B, num_diseases, attn_dim]
        queries = self.fc_query(final_disease_emb)
        
        # Compute keys and values from the image patch features.
        # keys, values: [B, N, attn_dim]
        keys = self.fc_key(image_patch_features)
        values = self.fc_value(image_patch_features)
        
        # Compute attention scores between each disease query and the image patches.
        # scores: [B, num_diseases, N]
        scores = torch.matmul(queries, keys.transpose(1, 2))
        scores = scores / math.sqrt(queries.size(-1))
        attn_weights = F.softmax(scores, dim=-1)
        
        # Compute attended image features per disease.
        # attended: [B, num_diseases, attn_dim]
        attended = torch.matmul(attn_weights, values)
        
        # Fuse the attended image features with the filtered disease embeddings.
        # fused: [B, num_diseases, attn_dim + disease_emb_dim]
        fused = torch.cat([attended, final_disease_emb], dim=-1)
        fused = self.ln_fused(fused)  # Apply layer normalization.
        # Compute per-disease location logits.
        # logits: [B, num_diseases, location_dim]
        logits = self.classifier(fused)
        
        return logits


class SeverityOrdinalClassifier(nn.Module):
    def __init__(self, img_dim, disease_emb_dim, attn_dim, hidden_dim, severity_dim, num_diseases):
        """
        Args:
            img_dim (int): Dimensionality of each image patch feature (e.g. 1024).
            disease_emb_dim (int): Dimensionality of the disease embedding.
            attn_dim (int): Dimensionality for the cross-attention projection.
            hidden_dim (int): Hidden layer size for fusion.
            severity_dim (int): Number of ordinal outputs (e.g. 5 thresholds).
            num_diseases (int): Total number of diseases.
        """
        super(SeverityOrdinalClassifier, self).__init__()
        # Projection layers for cross-attention.
        self.fc_key = Linear(img_dim, attn_dim)
        self.fc_value = Linear(img_dim, attn_dim)
        self.fc_query = Linear(disease_emb_dim, attn_dim)
        
        # Disease embeddings.
        self.disease_embed = nn.Embedding(num_diseases, disease_emb_dim)
        self.num_diseases = num_diseases
        
        # Fusion MLP: fuses attended image features with disease embeddings.
        self.fusion = nn.Sequential(
            nn.Linear(attn_dim + disease_emb_dim, hidden_dim),
            nn.ReLU()
        )
        # Instead of directly predicting 5 outputs, predict a single severity score.
        self.fc_severity = Linear(hidden_dim, 1)  # Output shape: [B, num_diseases, 1]
        
        # Learnable thresholds for the ordinal classifier.
        # We parameterize them via raw deltas that are made positive (via softplus)
        # and then cumulatively summed to guarantee monotonicity.
        self.delta = nn.Parameter(torch.zeros(severity_dim))  # severity_dim values.
        
        # This token will be used when a disease is absent.
        self.no_disease_token = nn.Parameter(torch.empty(disease_emb_dim))
        nn.init.xavier_uniform_(self.no_disease_token.unsqueeze(0))

        # Layer normalization applied after concatenation of attended features and disease embeddings.
        self.ln_fused = nn.LayerNorm(attn_dim + disease_emb_dim)
    
    def forward(self, image_patch_features, disease_logits=None, gold_disease_labels=None, threshold=0.5):
        """
        Args:
            image_patch_features: Tensor of shape [B, N, img_dim] (flattened image features).
            disease_logits: Optional tensor of shape [B, num_diseases] from a disease classifier.
            gold_disease_labels: Optional tensor with ground-truth disease labels (used for masking).
        Returns:
            Returns the computed ordinal logits of shape 
            [B, num_diseases, severity_dim].
        """
        B = image_patch_features.size(0)
        N = image_patch_features.size(1)
        # Get the standard disease embeddings: shape [B, num_diseases, disease_emb_dim].
        disease_emb = self.disease_embed.weight.unsqueeze(0).expand(B, -1, -1)
        # Expand the no-disease token to [B, num_diseases, disease_emb_dim].
        no_disease_emb = self.no_disease_token.unsqueeze(0).unsqueeze(0).expand(B, self.num_diseases, -1)
        if gold_disease_labels is not None:
            # Assume gold_disease_labels is a multi-hot vector.
            # Expand it to shape [B, num_diseases, 1].
            mask = gold_disease_labels.float().unsqueeze(-1)
            # Use the gold label as a gate: where 1, use disease_emb; where 0, use no_disease_emb.
            final_disease_emb = mask * disease_emb + (1 - mask) * no_disease_emb
        elif disease_logits is not None:
            # Use the disease logits to compute soft probabilities.
            prob = torch.sigmoid(disease_logits).unsqueeze(-1)  # [B, num_diseases, 1]
            # Derive a hard binary mask using the threshold.
            hard_mask = (prob > threshold).float()  # [B, num_diseases, 1]
            # Combine the two: if the hard mask indicates presence, use (disease_emb * prob),
            # else use the no_disease embedding.
            final_disease_emb = hard_mask * (disease_emb * prob) + (1 - hard_mask) * no_disease_emb
        else:
            warnings.warn("SeverityOrdinalClassifier: No filtering info provided; falling back to uniform embeddings.", UserWarning)
            final_disease_emb = self.disease_embed.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Project disease embeddings to query space.
        queries = self.fc_query(final_disease_emb)  # [B, num_diseases, attn_dim]
        
        # Compute keys and values from image patch features.
        keys = self.fc_key(image_patch_features)         # [B, N, attn_dim]
        values = self.fc_value(image_patch_features)       # [B, N, attn_dim]
        
        # Compute cross-attention: each disease attends to image patches.
        scores = torch.matmul(queries, keys.transpose(1, 2))  # [B, num_diseases, N]
        scores = scores / math.sqrt(queries.size(-1))
        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, values)  # [B, num_diseases, attn_dim]
        
        # Fuse attended image features with disease embeddings.
        fused = torch.cat([attended, final_disease_emb], dim=-1)  # [B, num_diseases, attn_dim + disease_emb_dim]
        # Apply layer normalization.
        fused = self.ln_fused(fused)
        fused_feat = self.fusion(fused)  # [B, num_diseases, hidden_dim]
        
        # Compute a single severity score per disease.
        severity_score = self.fc_severity(fused_feat)  # [B, num_diseases, 1]
        
        # Compute monotonic thresholds:
        # Use softplus to ensure positivity, then compute cumulative sum.
        thresholds = torch.cumsum(F.softplus(self.delta), dim=0)  # [severity_dim]
        
        # Broadcast severity_score and thresholds for subtraction.
        severity_score_exp = severity_score.expand(-1, -1, thresholds.size(0))  # [B, num_diseases, severity_dim]
        thresholds_exp = thresholds.unsqueeze(0).unsqueeze(0)  # [1, 1, severity_dim]
        
        # Compute ordinal logits: each element represents the logit for that binary decision.
        ordinal_logits = severity_score_exp - thresholds_exp  # [B, num_diseases, severity_dim]
        # After applying a sigmoid, these logits should yield probabilities that, when thresholded,
        # produce a cumulative binary vector (e.g., [1,0,0,0,0], [1,1,0,0,0], etc.)
        
        return ordinal_logits

