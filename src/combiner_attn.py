import torch
from torch import nn
import torch.nn.functional as F


class Combiner(nn.Module):
    """
    Combiner module which once trained fuses textual and visual information using attention.
    """

    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int, num_heads=4):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        :param num_heads: number of attention heads
        """
        super(Combiner, self).__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        # Create four layers of attention
        self.attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim=projection_dim, num_heads=num_heads) for _ in range(4)]
        )
        
        self.output_layer = nn.Linear(projection_dim, clip_feature_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.logit_scale = 100

    def forward(self, image_features: torch.tensor, text_features: torch.tensor,
                target_features: torch.tensor) -> torch.tensor:
        """
        Takes as input a triplet: image_features, text_features and target_features and outputs the logits which are
        the normalized dot product between the predicted features and the target_features.
        The logits are also multiplied by logit_scale parameter
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :param target_features: CLIP target image features
        :return: scaled logits
        """
        predicted_features = self.combine_features(image_features, text_features)
        target_features = F.normalize(target_features, dim=-1)

        logits = self.logit_scale * predicted_features @ target_features.T
        return logits

    def combine_features(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        """
        Combine the reference image features and the caption features using attention. It outputs the predicted features
        :param image_features: CLIP reference image features
        :param text_features: CLIP relative caption features
        :return: predicted features
        """
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        # Prepare for multi-head attention
        combined_features = torch.stack((text_projected_features, image_projected_features), dim=1)

        # Apply four layers of attention
        for attention_layer in self.attention_layers:
            attention_output, _ = attention_layer(combined_features, combined_features, combined_features)
            combined_features = attention_output  # Update combined_features for the next layer

        # We take the output corresponding to the text features
        output = attention_output[:, 0, :]  # Get the attention output related to text_features

        dynamic_scalar = self.dynamic_scalar(torch.cat((text_projected_features, image_projected_features), dim=-1))
        output = self.output_layer(output) + dynamic_scalar * text_features + (1 - dynamic_scalar) * image_features
        return F.normalize(output, dim=-1)