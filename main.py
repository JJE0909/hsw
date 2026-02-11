import os
import json
import time
import random
import string
import base64
import subprocess
import tempfile
import math
import re
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from curl_cffi import requests
from urllib.parse import urlparse

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import clip
from io import BytesIO


# ══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

class ChallengeType(Enum):
    IMAGE_CLASSIFICATION = "image_classification"      # "click on images containing X"
    BOUNDING_BOX = "bounding_box"                      # "click on the X in the image"
    MULTI_CHOICE = "multi_choice"                      # "which image matches X"
    DRAG_DROP = "drag_drop"                            # drag piece to correct position
    POINT_CLICK = "point_click"                        # click specific point on image
    GRID_CLASSIFICATION = "grid_classification"        # 3x3 or 4x4 grid selection
    UNKNOWN = "unknown"


@dataclass
class TaskResult:
    task_index: int
    task_key: str
    selected: bool = False
    confidence: float = 0.0
    click_point: Optional[Tuple[int, int]] = None
    drag_start: Optional[Tuple[int, int]] = None
    drag_end: Optional[Tuple[int, int]] = None
    grid_cells: List[int] = field(default_factory=list)
    similarity_score: float = 0.0
    label: str = ""


@dataclass
class SolutionResult:
    challenge_type: ChallengeType
    question: str
    tasks: List[TaskResult]
    overall_confidence: float = 0.0
    solved: bool = False
    annotated_image_path: str = ""
    answer_payload: Dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════
# CLIP AI SOLVER ENGINE
# ══════════════════════════════════════════════════════════════════════

class CLIPSolverEngine:
    """
    Multi-strategy CLIP-based solver for all hCaptcha challenge types.
    Uses OpenAI's ViT-L/14@336px — the largest and most accurate CLIP model.
    """

    PROMPT_TEMPLATES = [
        "a photo of a {}",
        "a photograph of a {}",
        "an image of a {}",
        "a picture of a {}",
        "a clear photo of a {}",
        "a photo showing a {}",
        "a real photo of a {}",
        "a bright photo of a {}",
        "a close-up photo of a {}",
        "a photo of a single {}",
        "a photo of a large {}",
        "a photo of a small {}",
        "a centered photo of a {}",
        "a good photo of a {}",
        "a photo of one {}",
        "a photo of many {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a dark photo of a {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a blurry photo of a {}",
        "a jpeg photo of a {}",
        "a pixelated photo of a {}",
        "a photo containing a {}",
        "a photo with a {} in it",
        "a {} in a photo",
        "a {} in this image",
        "this image contains a {}",
        "a photo that shows a {}",
        "a photo depicting a {}",
    ]

    NEGATIVE_TEMPLATES = [
        "a photo without a {}",
        "a photo that does not contain a {}",
        "an image with no {}",
        "a picture lacking a {}",
        "a photo of something else, not a {}",
        "an empty photo with no {} present",
    ]

    # Common hCaptcha question patterns and how to extract the target
    QUESTION_PATTERNS = [
        (r"(?:please\s+)?(?:click|select|choose|pick)\s+(?:on\s+)?(?:each|all|the|every)?\s*(?:images?|photos?|pictures?)?\s*(?:that\s+)?(?:contains?|containing|showing|with|of|depicting|that\s+(?:has|have|show|depicts?))\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\s*\.?\s*$)", None),
        (r"(?:please\s+)?(?:click|select|choose|pick)\s+(?:on\s+)?(?:each|all|the|every)?\s*(?:images?|photos?|pictures?)?\s*(?:of|with)\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\s*\.?\s*$)", None),
        (r"(?:which\s+)?(?:images?|photos?|pictures?)\s+(?:contains?|shows?|has|have|depicts?)\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\s*\??\.?\s*$)", None),
        (r"(?:select|click|choose)\s+(?:the\s+)?(.+?)(?:\s*\.?\s*$)", None),
        (r"(?:find|identify|locate)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+in\s+(?:the|this)\s+image)?(?:\s*\.?\s*$)", None),
        (r"(?:drag|move)\s+(?:the\s+)?(.+?)\s+(?:to|into|onto)\s+(?:the\s+|its?\s+)?(?:correct\s+)?(.+?)(?:\s*\.?\s*$)", "drag"),
        (r"(?:click|point|tap)\s+(?:on\s+)?(?:the\s+)?(.+?)(?:\s+in\s+(?:the|this)\s+image)?(?:\s*\.?\s*$)", "point"),
    ]

    # Semantic expansions for common hCaptcha targets
    SEMANTIC_EXPANSIONS = {
        "airplane": ["airplane", "aeroplane", "aircraft", "jet", "plane", "airliner", "aviation"],
        "bicycle": ["bicycle", "bike", "cycle", "two-wheeler", "pedal bike", "mountain bike", "road bike"],
        "boat": ["boat", "ship", "vessel", "watercraft", "sailboat", "yacht", "canoe", "kayak", "motorboat"],
        "bus": ["bus", "autobus", "coach", "shuttle bus", "city bus", "school bus", "double-decker bus"],
        "car": ["car", "automobile", "vehicle", "sedan", "SUV", "hatchback", "coupe", "convertible"],
        "cat": ["cat", "kitten", "feline", "tabby cat", "house cat", "domestic cat"],
        "chair": ["chair", "seat", "armchair", "office chair", "dining chair", "wooden chair", "rocking chair"],
        "dog": ["dog", "puppy", "canine", "hound", "pet dog", "domestic dog"],
        "horse": ["horse", "stallion", "mare", "pony", "equine", "foal"],
        "motorcycle": ["motorcycle", "motorbike", "bike", "chopper", "scooter", "moped"],
        "person": ["person", "human", "man", "woman", "people", "individual", "pedestrian", "figure"],
        "train": ["train", "locomotive", "railway", "railroad", "metro", "subway", "tram"],
        "truck": ["truck", "lorry", "pickup", "semi-truck", "delivery truck", "cargo truck"],
        "traffic light": ["traffic light", "traffic signal", "stoplight", "signal light", "red light", "green light"],
        "fire hydrant": ["fire hydrant", "hydrant", "fire plug", "street hydrant"],
        "stop sign": ["stop sign", "stop signal", "road sign"],
        "parking meter": ["parking meter", "meter", "parking machine"],
        "bird": ["bird", "avian", "sparrow", "pigeon", "eagle", "songbird", "flying bird"],
        "elephant": ["elephant", "pachyderm", "tusker"],
        "bear": ["bear", "grizzly", "polar bear", "brown bear", "teddy bear"],
        "zebra": ["zebra", "striped horse"],
        "giraffe": ["giraffe", "tall animal with long neck"],
        "umbrella": ["umbrella", "parasol", "sunshade"],
        "handbag": ["handbag", "purse", "bag", "tote", "clutch"],
        "tie": ["tie", "necktie", "bow tie", "cravat"],
        "suitcase": ["suitcase", "luggage", "travel bag", "briefcase"],
        "frisbee": ["frisbee", "flying disc", "disc"],
        "skis": ["skis", "ski", "skiing equipment"],
        "snowboard": ["snowboard", "snow board"],
        "sports ball": ["sports ball", "ball", "soccer ball", "football", "basketball", "tennis ball", "baseball"],
        "kite": ["kite", "flying kite"],
        "baseball bat": ["baseball bat", "bat", "wooden bat"],
        "skateboard": ["skateboard", "board", "skate"],
        "surfboard": ["surfboard", "surf board", "board"],
        "tennis racket": ["tennis racket", "racket", "racquet"],
        "bottle": ["bottle", "water bottle", "glass bottle", "plastic bottle"],
        "wine glass": ["wine glass", "glass", "goblet", "champagne glass"],
        "cup": ["cup", "mug", "coffee cup", "tea cup"],
        "fork": ["fork", "dinner fork", "utensil"],
        "knife": ["knife", "blade", "kitchen knife", "butter knife"],
        "spoon": ["spoon", "tablespoon", "teaspoon"],
        "bowl": ["bowl", "dish", "soup bowl", "cereal bowl"],
        "banana": ["banana", "yellow banana", "fruit banana"],
        "apple": ["apple", "red apple", "green apple", "fruit"],
        "sandwich": ["sandwich", "sub", "hoagie"],
        "orange": ["orange", "orange fruit", "citrus"],
        "broccoli": ["broccoli", "green vegetable"],
        "carrot": ["carrot", "orange vegetable"],
        "hot dog": ["hot dog", "hotdog", "frankfurter", "sausage"],
        "pizza": ["pizza", "pie", "pizza pie", "slice of pizza"],
        "donut": ["donut", "doughnut", "pastry"],
        "cake": ["cake", "birthday cake", "pastry", "dessert"],
        "couch": ["couch", "sofa", "loveseat", "settee"],
        "bed": ["bed", "mattress", "bedroom"],
        "toilet": ["toilet", "lavatory", "bathroom", "restroom"],
        "tv": ["tv", "television", "monitor", "screen", "display"],
        "laptop": ["laptop", "notebook", "computer", "portable computer"],
        "mouse": ["mouse", "computer mouse", "wireless mouse"],
        "keyboard": ["keyboard", "computer keyboard", "typing device"],
        "cell phone": ["cell phone", "mobile phone", "smartphone", "phone", "cellphone", "iphone", "android phone"],
        "microwave": ["microwave", "microwave oven", "kitchen appliance"],
        "oven": ["oven", "stove", "range", "cooking appliance"],
        "toaster": ["toaster", "bread toaster"],
        "sink": ["sink", "kitchen sink", "bathroom sink", "basin"],
        "refrigerator": ["refrigerator", "fridge", "icebox", "cooler"],
        "book": ["book", "textbook", "novel", "paperback", "hardcover"],
        "clock": ["clock", "timepiece", "wall clock", "alarm clock"],
        "vase": ["vase", "flower vase", "decorative vase"],
        "scissors": ["scissors", "shears", "cutting tool"],
        "teddy bear": ["teddy bear", "stuffed bear", "plush toy", "stuffed animal"],
        "toothbrush": ["toothbrush", "tooth brush", "dental brush"],
        "hair drier": ["hair drier", "hair dryer", "blow dryer"],
        # hCaptcha specific
        "motorbus": ["motorbus", "bus", "autobus", "coach", "city bus", "public bus", "transit bus"],
        "seaplane": ["seaplane", "float plane", "flying boat", "amphibious aircraft", "water plane"],
        "living room": ["living room", "lounge", "sitting room", "family room", "den"],
        "bedroom": ["bedroom", "sleeping room", "bed room"],
        "kitchen": ["kitchen", "cooking area", "galley"],
        "bathroom": ["bathroom", "restroom", "washroom", "lavatory"],
        "office": ["office", "workspace", "study", "workroom"],
        "swimming pool": ["swimming pool", "pool", "swimming area"],
        "bridge": ["bridge", "overpass", "viaduct", "footbridge"],
        "church": ["church", "chapel", "cathedral", "temple"],
        "tower": ["tower", "turret", "skyscraper", "spire"],
        "mountain": ["mountain", "peak", "summit", "hill", "mountain range"],
        "forest": ["forest", "woods", "woodland", "trees"],
        "river": ["river", "stream", "creek", "waterway"],
        "lake": ["lake", "pond", "reservoir"],
        "ocean": ["ocean", "sea", "water body"],
        "beach": ["beach", "shore", "coast", "seaside", "sandy beach"],
        "desert": ["desert", "sand dunes", "arid landscape"],
        "garden": ["garden", "yard", "park", "botanical garden"],
        "road": ["road", "street", "highway", "path", "lane"],
        "crosswalk": ["crosswalk", "pedestrian crossing", "zebra crossing"],
        "staircase": ["staircase", "stairs", "steps", "stairway"],
        "lion": ["lion", "big cat", "feline", "lioness", "male lion"],
        "tiger": ["tiger", "big cat", "striped cat"],
        "monkey": ["monkey", "primate", "ape", "chimpanzee"],
        "penguin": ["penguin", "arctic bird", "flightless bird"],
        "dolphin": ["dolphin", "porpoise", "marine mammal"],
        "whale": ["whale", "marine mammal", "humpback"],
        "fish": ["fish", "marine animal", "aquatic creature"],
        "butterfly": ["butterfly", "moth", "insect", "lepidoptera"],
        "flower": ["flower", "bloom", "blossom", "floral"],
        "tree": ["tree", "oak", "pine", "deciduous tree", "plant"],
        "grass": ["grass", "lawn", "turf", "greenery"],
        "rock": ["rock", "stone", "boulder", "pebble"],
        "cloud": ["cloud", "cumulus", "sky", "overcast"],
        "sun": ["sun", "sunshine", "sunlight", "solar"],
        "moon": ["moon", "lunar", "crescent"],
        "star": ["star", "stars", "celestial"],
        "hat": ["hat", "cap", "headwear", "headgear", "beanie"],
        "shoe": ["shoe", "footwear", "sneaker", "boot", "sandal"],
        "shirt": ["shirt", "top", "blouse", "t-shirt", "polo"],
        "dress": ["dress", "gown", "frock"],
        "glasses": ["glasses", "eyeglasses", "spectacles", "sunglasses"],
        "watch": ["watch", "wristwatch", "timepiece"],
        "ring": ["ring", "jewelry", "band"],
        "necklace": ["necklace", "chain", "pendant", "jewelry"],
    }

    def __init__(self, device: str = None, model_name: str = "ViT-L/14@336px"):
        """
        Initialize CLIP solver with the specified model.

        Available models (in order of accuracy):
        - ViT-L/14@336px  (best accuracy, largest)
        - ViT-L/14
        - ViT-B/32 (fastest, least accurate)
        - ViT-B/16
        - RN50x64
        - RN50x16
        - RN50x4
        - RN101
        - RN50
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model_name = model_name

        print(f"\n{'='*60}")
        print(f"  CLIP AI Solver Engine")
        print(f"{'='*60}")
        print(f"  Model: {model_name}")
        print(f"  Device: {device}")
        print(f"  Loading model...")

        t0 = time.time()
        self.model, self.preprocess = clip.load(model_name, device=device, jit=False)
        self.model.eval()

        dt = time.time() - t0
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  ✓ Loaded in {dt:.1f}s ({total_params/1e6:.0f}M parameters)")
        print(f"  Input resolution: {self.model.visual.input_resolution}")
        print(f"{'='*60}\n")

        # Cache for text embeddings
        self._text_cache = {}

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a single PIL image to CLIP embedding."""
        img_input = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(img_input)
        features = F.normalize(features, dim=-1)
        return features

    @torch.no_grad()
    def encode_images_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode multiple PIL images to CLIP embeddings in a single batch."""
        if not images:
            return torch.empty(0, 512).to(self.device)

        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        features = self.model.encode_image(batch)
        features = F.normalize(features, dim=-1)
        return features

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text prompts to CLIP embeddings."""
        cache_key = tuple(sorted(texts))
        if cache_key in self._text_cache:
            return self._text_cache[cache_key]

        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        features = self.model.encode_text(tokens)
        features = F.normalize(features, dim=-1)
        self._text_cache[cache_key] = features
        return features

    def parse_question(self, question: str) -> Tuple[str, Optional[str]]:
        """
        Parse the hCaptcha question to extract the target object/concept.
        Returns (target, challenge_hint) where challenge_hint indicates special type.
        """
        question = question.strip().lower()
        question = re.sub(r'\s+', ' ', question)

        # Remove common prefixes
        question = re.sub(r'^please\s+', '', question)

        for pattern, hint in self.QUESTION_PATTERNS:
            m = re.search(pattern, question, re.IGNORECASE)
            if m:
                target = m.group(1).strip().rstrip('.?!')
                # Clean up articles
                target = re.sub(r'^(a|an|the)\s+', '', target)
                return target, hint

        # Fallback: return the whole question cleaned up
        cleaned = re.sub(r'^(select|click|choose|pick|find|identify)\s+(on\s+)?', '', question)
        cleaned = re.sub(r'\s*(images?|photos?|pictures?)\s*(of|with|containing|showing|that)?\s*', ' ', cleaned)
        cleaned = cleaned.strip().rstrip('.?!')
        cleaned = re.sub(r'^(a|an|the)\s+', '', cleaned)
        return cleaned if cleaned else question, None

    def build_prompts(self, target: str) -> Tuple[List[str], List[str]]:
        """
        Build comprehensive positive and negative prompts for the target.
        Uses semantic expansions and multiple templates for robustness.
        """
        # Get semantic expansions
        target_lower = target.lower().strip()
        expansions = self.SEMANTIC_EXPANSIONS.get(target_lower, [target_lower])

        # Always include the original target
        if target_lower not in expansions:
            expansions = [target_lower] + expansions

        # Build positive prompts with all templates × expansions
        positive = []
        for expansion in expansions:
            for template in self.PROMPT_TEMPLATES:
                positive.append(template.format(expansion))

        # Build negative prompts
        negative = []
        for template in self.NEGATIVE_TEMPLATES:
            negative.append(template.format(target_lower))

        # Add generic negatives
        negative.extend([
            "a photo of something else",
            "a random photo",
            "an empty image",
            "a photo with no distinguishable objects",
            "a blurry unrecognizable photo",
            "a photo of a completely different object",
            "nothing relevant in this photo",
        ])

        return positive, negative

    def compute_similarity_ensemble(
        self,
        image_features: torch.Tensor,
        positive_prompts: List[str],
        negative_prompts: List[str],
    ) -> Tuple[float, float]:
        """
        Compute similarity score using ensemble of prompts.
        Returns (positive_score, confidence) where score is in [0, 1].
        """
        pos_features = self.encode_text(positive_prompts)
        neg_features = self.encode_text(negative_prompts)

        # Average positive and negative embeddings for robustness
        pos_mean = F.normalize(pos_features.mean(dim=0, keepdim=True), dim=-1)
        neg_mean = F.normalize(neg_features.mean(dim=0, keepdim=True), dim=-1)

        # Compute cosine similarities
        pos_sim = (image_features @ pos_mean.T).item()
        neg_sim = (image_features @ neg_mean.T).item()

        # Also compute max similarity across individual prompts for peak detection
        all_pos_sims = (image_features @ pos_features.T).squeeze()
        if all_pos_sims.dim() == 0:
            max_pos_sim = all_pos_sims.item()
        else:
            max_pos_sim = all_pos_sims.max().item()

        # Weighted combination: mean similarity + peak boost
        combined_pos = 0.7 * pos_sim + 0.3 * max_pos_sim

        # Softmax-style scoring between positive and negative
        temperature = 0.01  # Lower = more decisive
        logits = torch.tensor([combined_pos / temperature, neg_sim / temperature])
        probs = F.softmax(logits, dim=0)
        score = probs[0].item()

        # Confidence based on separation
        confidence = abs(combined_pos - neg_sim)

        return score, confidence

    def solve_image_classification(
        self,
        images: List[Image.Image],
        task_keys: List[str],
        target: str,
        threshold: float = 0.5,
    ) -> List[TaskResult]:
        """
        Solve image classification challenges:
        "Select all images containing X"
        """
        print(f"\n  [CLIP] Solving classification: '{target}'")
        print(f"  [CLIP] {len(images)} images to evaluate")

        positive_prompts, negative_prompts = self.build_prompts(target)
        print(f"  [CLIP] {len(positive_prompts)} positive, {len(negative_prompts)} negative prompts")

        # Encode all images in batch
        image_features = self.encode_images_batch(images)

        results = []
        scores = []

        for i in range(len(images)):
            feat = image_features[i:i+1]
            score, confidence = self.compute_similarity_ensemble(
                feat, positive_prompts, negative_prompts
            )
            scores.append(score)

            result = TaskResult(
                task_index=i,
                task_key=task_keys[i] if i < len(task_keys) else f"task_{i}",
                similarity_score=score,
                confidence=confidence,
                label=target,
            )
            results.append(result)

        # Adaptive thresholding: use Otsu-like method on scores
        scores_array = np.array(scores)
        adaptive_threshold = self._compute_adaptive_threshold(scores_array, default=threshold)

        print(f"\n  [CLIP] Score distribution:")
        print(f"         Min: {scores_array.min():.4f}")
        print(f"         Max: {scores_array.max():.4f}")
        print(f"         Mean: {scores_array.mean():.4f}")
        print(f"         Std: {scores_array.std():.4f}")
        print(f"         Threshold: {adaptive_threshold:.4f}")

        for i, result in enumerate(results):
            result.selected = scores[i] >= adaptive_threshold
            status = "✓ MATCH" if result.selected else "✗ skip"
            print(f"    [{i:2d}] {status}  score={scores[i]:.4f}  conf={result.confidence:.4f}")

        return results

    def solve_point_click(
        self,
        image: Image.Image,
        task_key: str,
        target: str,
        grid_size: int = 16,
    ) -> TaskResult:
        """
        Solve point-click challenges by finding the target location in the image.
        Uses a sliding window / grid approach with CLIP.
        """
        print(f"\n  [CLIP] Solving point-click: '{target}'")

        w, h = image.size
        positive_prompts, negative_prompts = self.build_prompts(target)

        # Create overlapping patches at multiple scales
        patches = []
        patch_centers = []
        patch_sizes = [
            (w // 2, h // 2),    # large patches
            (w // 3, h // 3),    # medium patches
            (w // 4, h // 4),    # small patches
            (w // 5, h // 5),    # smaller patches
        ]

        for pw, ph in patch_sizes:
            stride_x = max(pw // 3, 1)
            stride_y = max(ph // 3, 1)

            for y in range(0, h - ph + 1, stride_y):
                for x in range(0, w - pw + 1, stride_x):
                    patch = image.crop((x, y, x + pw, y + ph))
                    patches.append(patch)
                    patch_centers.append((x + pw // 2, y + ph // 2))

        print(f"  [CLIP] Evaluating {len(patches)} patches...")

        # Process in batches to avoid OOM
        batch_size = 64
        all_scores = []

        for start in range(0, len(patches), batch_size):
            end = min(start + batch_size, len(patches))
            batch_patches = patches[start:end]
            features = self.encode_images_batch(batch_patches)

            for j in range(features.shape[0]):
                score, _ = self.compute_similarity_ensemble(
                    features[j:j+1], positive_prompts, negative_prompts
                )
                all_scores.append(score)

        # Find best patch center using weighted average of top-k patches
        scores_array = np.array(all_scores)
        top_k = min(10, len(scores_array))
        top_indices = np.argsort(scores_array)[-top_k:]
        top_scores = scores_array[top_indices]

        # Softmax weighting
        weights = np.exp(top_scores * 10)
        weights /= weights.sum()

        cx = sum(patch_centers[idx][0] * w for idx, w in zip(top_indices, weights))
        cy = sum(patch_centers[idx][1] * w for idx, w in zip(top_indices, weights))

        click_x = int(np.clip(cx, 5, w - 5))
        click_y = int(np.clip(cy, 5, h - 5))

        best_score = scores_array.max()
        print(f"  [CLIP] Best click point: ({click_x}, {click_y}) score={best_score:.4f}")

        return TaskResult(
            task_index=0,
            task_key=task_key,
            selected=True,
            confidence=best_score,
            click_point=(click_x, click_y),
            similarity_score=best_score,
            label=target,
        )

    def solve_bounding_box(
        self,
        image: Image.Image,
        task_key: str,
        target: str,
        grid_divisions: int = 8,
    ) -> TaskResult:
        """
        Solve bounding box challenges by localizing the target object.
        Returns a click point at the center of the detected region.
        """
        print(f"\n  [CLIP] Solving bounding box: '{target}'")
        # Essentially same as point_click but with denser grid
        return self.solve_point_click(image, task_key, target, grid_size=grid_divisions)

    def solve_drag_drop(
        self,
        main_image: Image.Image,
        piece_image: Image.Image,
        task_key: str,
        target: str = "",
        grid_divisions: int = 12,
    ) -> TaskResult:
        """
        Solve drag-and-drop challenges by finding where the piece fits.
        Compares the piece against patches of the main image.
        """
        print(f"\n  [CLIP] Solving drag-drop")

        w, h = main_image.size
        pw, ph = piece_image.size

        # Encode the piece
        piece_features = self.encode_image(piece_image)

        # Slide the piece-sized window across the main image
        patches = []
        positions = []
        stride = max(min(pw, ph) // 4, 4)

        for y in range(0, h - ph + 1, stride):
            for x in range(0, w - pw + 1, stride):
                patch = main_image.crop((x, y, x + pw, y + ph))
                patches.append(patch)
                positions.append((x + pw // 2, y + ph // 2))

        print(f"  [CLIP] Evaluating {len(patches)} positions...")

        # Process in batches
        batch_size = 64
        all_sims = []

        for start in range(0, len(patches), batch_size):
            end = min(start + batch_size, len(patches))
            batch_patches = patches[start:end]
            features = self.encode_images_batch(batch_patches)
            sims = (features @ piece_features.T).squeeze(-1)
            all_sims.extend(sims.cpu().tolist())

        sims_array = np.array(all_sims)
        best_idx = int(np.argmax(sims_array))
        best_pos = positions[best_idx]
        best_score = sims_array[best_idx]

        # Drag start = center of piece image (relative coords)
        drag_start = (pw // 2, ph // 2)
        drag_end = best_pos

        print(f"  [CLIP] Best drop position: {drag_end} score={best_score:.4f}")

        return TaskResult(
            task_index=0,
            task_key=task_key,
            selected=True,
            confidence=float(best_score),
            drag_start=drag_start,
            drag_end=drag_end,
            similarity_score=float(best_score),
            label=target or "puzzle piece",
        )

    def solve_grid_classification(
        self,
        image: Image.Image,
        task_key: str,
        target: str,
        grid_rows: int = 3,
        grid_cols: int = 3,
        threshold: float = 0.5,
    ) -> TaskResult:
        """
        Solve grid-based classification (e.g., 3x3 grid where you select cells containing X).
        """
        print(f"\n  [CLIP] Solving grid {grid_rows}x{grid_cols}: '{target}'")

        w, h = image.size
        cell_w = w // grid_cols
        cell_h = h // grid_rows

        cells = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                x1 = col * cell_w
                y1 = row * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h
                cell = image.crop((x1, y1, x2, y2))
                cells.append(cell)

        positive_prompts, negative_prompts = self.build_prompts(target)
        features = self.encode_images_batch(cells)

        selected_cells = []
        scores = []

        for i in range(len(cells)):
            score, _ = self.compute_similarity_ensemble(
                features[i:i+1], positive_prompts, negative_prompts
            )
            scores.append(score)

        scores_array = np.array(scores)
        adaptive_threshold = self._compute_adaptive_threshold(scores_array, default=threshold)

        for i, score in enumerate(scores):
            row, col = divmod(i, grid_cols)
            if score >= adaptive_threshold:
                selected_cells.append(i)
                print(f"    Cell [{row},{col}] ✓ MATCH  score={score:.4f}")
            else:
                print(f"    Cell [{row},{col}] ✗ skip   score={score:.4f}")

        return TaskResult(
            task_index=0,
            task_key=task_key,
            selected=len(selected_cells) > 0,
            confidence=float(scores_array.max()),
            grid_cells=selected_cells,
            similarity_score=float(scores_array.mean()),
            label=target,
        )

    def _compute_adaptive_threshold(
        self, scores: np.ndarray, default: float = 0.5
    ) -> float:
        """
        Compute an adaptive threshold using a combination of methods:
        1. Otsu's method on the score distribution
        2. Gap analysis (largest gap in sorted scores)
        3. Statistical method (mean + fraction of std)
        """
        if len(scores) < 3:
            return default

        sorted_scores = np.sort(scores)

        # Method 1: Gap analysis
        gaps = np.diff(sorted_scores)
        if len(gaps) > 0:
            max_gap_idx = np.argmax(gaps)
            gap_threshold = (sorted_scores[max_gap_idx] + sorted_scores[max_gap_idx + 1]) / 2
            max_gap = gaps[max_gap_idx]
        else:
            gap_threshold = default
            max_gap = 0

        # Method 2: Statistical
        mean = scores.mean()
        std = scores.std()
        stat_threshold = mean + 0.3 * std if std > 0.01 else mean

        # Method 3: Percentile-based
        p60 = np.percentile(scores, 60)

        # Combine methods with weighting based on score distribution
        if max_gap > 0.05:
            # Clear separation exists — trust gap analysis
            threshold = 0.6 * gap_threshold + 0.2 * stat_threshold + 0.2 * p60
        elif std > 0.05:
            # Some spread — use statistical
            threshold = 0.3 * gap_threshold + 0.5 * stat_threshold + 0.2 * p60
        else:
            # Tight distribution — use a more generous threshold
            threshold = 0.2 * gap_threshold + 0.3 * stat_threshold + 0.5 * p60

        # Clamp to reasonable range
        threshold = np.clip(threshold, 0.35, 0.85)

        return float(threshold)

    @torch.no_grad()
    def zero_shot_classify(
        self,
        image: Image.Image,
        candidate_labels: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Zero-shot image classification with arbitrary labels.
        Returns sorted list of (label, probability) pairs.
        """
        image_features = self.encode_image(image)

        prompts = [f"a photo of a {label}" for label in candidate_labels]
        text_features = self.encode_text(prompts)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = similarity.squeeze().cpu().numpy()

        results = list(zip(candidate_labels, probs))
        results.sort(key=lambda x: x[1], reverse=True)
        return results


# ══════════════════════════════════════════════════════════════════════
# SOLUTION ANNOTATOR — Visual overlay rendering
# ══════════════════════════════════════════════════════════════════════

class SolutionAnnotator:
    """
    Creates annotated solution images with:
    - Green overlays for drag targets / selected images
    - Red circles for click points
    - Confidence scores overlaid
    - Grid cell highlights
    - Composite overview images
    """

    # Color scheme
    GREEN = (0, 200, 0)
    GREEN_TRANSPARENT = (0, 200, 0, 80)
    GREEN_BORDER = (0, 255, 0)
    RED = (220, 40, 40)
    RED_TRANSPARENT = (220, 40, 40, 60)
    RED_BORDER = (255, 60, 60)
    BLUE = (40, 120, 220)
    BLUE_TRANSPARENT = (40, 120, 220, 80)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    YELLOW = (255, 220, 0)
    GRAY = (128, 128, 128)
    DARK_OVERLAY = (0, 0, 0, 120)

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Try to load a decent font
        self.font_large = self._load_font(24)
        self.font_medium = self._load_font(16)
        self.font_small = self._load_font(12)
        self.font_tiny = self._load_font(10)

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Try to load a TrueType font, fall back to default."""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSMono.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    return ImageFont.truetype(fp, size)
                except:
                    continue
        try:
            return ImageFont.truetype("arial.ttf", size)
        except:
            return ImageFont.load_default()

    def annotate_classification(
        self,
        images: List[Image.Image],
        results: List[TaskResult],
        question: str,
        challenge_type: ChallengeType,
    ) -> str:
        """
        Create annotated composite image for classification challenges.
        Selected images get green overlay, rejected get red tint.
        """
        n = len(images)
        if n == 0:
            return ""

        # Determine grid layout
        cols = min(n, 4) if n > 1 else 1
        rows = math.ceil(n / cols)

        # Standardize image sizes
        cell_size = 256
        padding = 8
        header_height = 80
        footer_height = 40

        total_w = cols * (cell_size + padding) + padding
        total_h = header_height + rows * (cell_size + padding) + padding + footer_height

        canvas = Image.new("RGBA", (total_w, total_h), (30, 30, 30, 255))
        draw = ImageDraw.Draw(canvas)

        # Header with question
        draw.rectangle([(0, 0), (total_w, header_height)], fill=(20, 20, 20, 255))
        wrapped = self._wrap_text(question, total_w - 20, self.font_medium)
        y_text = 10
        for line in wrapped[:3]:
            draw.text((10, y_text), line, fill=self.WHITE, font=self.font_medium)
            y_text += 20

        # Draw each image cell
        selected_count = sum(1 for r in results if r.selected)

        for i, (img, result) in enumerate(zip(images, results)):
            row = i // cols
            col = i % cols

            x = padding + col * (cell_size + padding)
            y = header_height + padding + row * (cell_size + padding)

            # Resize image to cell size
            cell_img = img.copy().convert("RGBA")
            cell_img = cell_img.resize((cell_size, cell_size), Image.LANCZOS)

            if result.selected:
                # Green overlay for selected
                overlay = Image.new("RGBA", (cell_size, cell_size), self.GREEN_TRANSPARENT)
                cell_img = Image.alpha_composite(cell_img, overlay)

                # Green border (thick)
                border_draw = ImageDraw.Draw(cell_img)
                for b in range(4):
                    border_draw.rectangle(
                        [(b, b), (cell_size - 1 - b, cell_size - 1 - b)],
                        outline=self.GREEN_BORDER
                    )

                # Checkmark
                self._draw_checkmark(border_draw, cell_size - 35, 10, 20, self.GREEN_BORDER)

            else:
                # Slight red/dark tint for rejected
                overlay = Image.new("RGBA", (cell_size, cell_size), self.RED_TRANSPARENT)
                cell_img = Image.alpha_composite(cell_img, overlay)

                # Gray border
                border_draw = ImageDraw.Draw(cell_img)
                for b in range(2):
                    border_draw.rectangle(
                        [(b, b), (cell_size - 1 - b, cell_size - 1 - b)],
                        outline=self.GRAY
                    )

                # X mark
                self._draw_xmark(border_draw, cell_size - 30, 15, 15, self.RED)

            # Score label
            score_text = f"{result.similarity_score:.3f}"
            label_bg_color = self.GREEN if result.selected else self.RED
            self._draw_label(
                ImageDraw.Draw(cell_img),
                5, cell_size - 25,
                score_text,
                bg_color=label_bg_color,
                font=self.font_small,
            )

            # Index label
            self._draw_label(
                ImageDraw.Draw(cell_img),
                5, 5,
                f"#{i}",
                bg_color=(60, 60, 60),
                font=self.font_small,
            )

            canvas.paste(cell_img, (x, y))

        # Footer
        footer_y = total_h - footer_height
        draw.rectangle([(0, footer_y), (total_w, total_h)], fill=(20, 20, 20, 255))
        summary = f"Selected: {selected_count}/{n} | Type: {challenge_type.value}"
        draw.text((10, footer_y + 10), summary, fill=self.YELLOW, font=self.font_small)

        # Save
        out_path = os.path.join(self.output_dir, "solution_annotated.png")
        canvas.save(out_path, "PNG")
        print(f"\n  [Annotator] Saved: {out_path}")
        return out_path

    def annotate_point_click(
        self,
        image: Image.Image,
        result: TaskResult,
        question: str,
    ) -> str:
        """
        Create annotated image with red circle at click point.
        """
        annotated = image.copy().convert("RGBA")
        draw = ImageDraw.Draw(annotated)

        if result.click_point:
            cx, cy = result.click_point

            # Outer glow circle (larger, semi-transparent)
            for r in range(40, 15, -5):
                alpha = int(40 * (40 - r) / 25)
                color = (255, 60, 60, alpha)
                glow = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
                glow_draw = ImageDraw.Draw(glow)
                glow_draw.ellipse(
                    [(cx - r, cy - r), (cx + r, cy + r)],
                    outline=color,
                    width=2,
                )
                annotated = Image.alpha_composite(annotated, glow)
                draw = ImageDraw.Draw(annotated)

            # Main red circle
            radius = 18
            draw.ellipse(
                [(cx - radius, cy - radius), (cx + radius, cy + radius)],
                outline=self.RED,
                width=3,
            )

            # Inner dot
            draw.ellipse(
                [(cx - 4, cy - 4), (cx + 4, cy + 4)],
                fill=self.RED,
            )

            # Crosshair lines
            line_len = 30
            draw.line([(cx - line_len, cy), (cx - radius - 3, cy)], fill=self.RED, width=2)
            draw.line([(cx + radius + 3, cy), (cx + line_len, cy)], fill=self.RED, width=2)
            draw.line([(cx, cy - line_len), (cx, cy - radius - 3)], fill=self.RED, width=2)
            draw.line([(cx, cy + radius + 3), (cx, cy + line_len)], fill=self.RED, width=2)

            # Coordinate label
            coord_text = f"({cx}, {cy})"
            self._draw_label(draw, cx + 25, cy - 20, coord_text, bg_color=self.RED, font=self.font_small)

            # Confidence label
            conf_text = f"conf: {result.confidence:.3f}"
            self._draw_label(draw, cx + 25, cy + 5, conf_text, bg_color=(60, 60, 60), font=self.font_tiny)

        # Question overlay at top
        self._draw_question_bar(draw, question, annotated.size[0])

        out_path = os.path.join(self.output_dir, "solution_click.png")
        annotated.save(out_path, "PNG")
        print(f"\n  [Annotator] Saved: {out_path}")
        return out_path

    def annotate_drag_drop(
        self,
        main_image: Image.Image,
        piece_image: Optional[Image.Image],
        result: TaskResult,
        question: str,
    ) -> str:
        """
        Create annotated image with green overlay at drag target.
        Shows drag start (blue) and end (green) with arrow.
        """
        annotated = main_image.copy().convert("RGBA")
        draw = ImageDraw.Draw(annotated)

        if result.drag_end:
            ex, ey = result.drag_end

            # Green target zone overlay
            if piece_image:
                pw, ph = piece_image.size
            else:
                pw, ph = 50, 50

            half_w, half_h = pw // 2, ph // 2

            # Semi-transparent green rectangle at target
            target_overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
            target_draw = ImageDraw.Draw(target_overlay)
            target_draw.rectangle(
                [(ex - half_w, ey - half_h), (ex + half_w, ey + half_h)],
                fill=(0, 200, 0, 80),
                outline=(0, 255, 0),
                width=3,
            )
            annotated = Image.alpha_composite(annotated, target_overlay)
            draw = ImageDraw.Draw(annotated)

            # Target center marker
            draw.ellipse(
                [(ex - 6, ey - 6), (ex + 6, ey + 6)],
                fill=self.GREEN,
                outline=self.GREEN_BORDER,
                width=2,
            )

            # If we have drag start, draw arrow
            if result.drag_start:
                sx, sy = result.drag_start
                # Draw arrow from start to end
                self._draw_arrow(draw, sx, sy, ex, ey, color=self.YELLOW, width=3)

                # Blue circle at start
                draw.ellipse(
                    [(sx - 10, sy - 10), (sx + 10, sy + 10)],
                    outline=self.BLUE,
                    width=3,
                )
                self._draw_label(draw, sx + 15, sy - 10, "START", bg_color=self.BLUE, font=self.font_tiny)

            # Label at target
            self._draw_label(
                draw, ex + half_w + 5, ey - 10,
                f"DROP ({ex},{ey})",
                bg_color=self.GREEN,
                font=self.font_small,
            )

        # Question overlay
        self._draw_question_bar(draw, question, annotated.size[0])

        out_path = os.path.join(self.output_dir, "solution_drag.png")
        annotated.save(out_path, "PNG")
        print(f"\n  [Annotator] Saved: {out_path}")
        return out_path

    def annotate_grid(
        self,
        image: Image.Image,
        result: TaskResult,
        question: str,
        grid_rows: int = 3,
        grid_cols: int = 3,
    ) -> str:
        """
        Create annotated grid image with selected cells highlighted green.
        """
        annotated = image.copy().convert("RGBA")
        w, h = annotated.size
        cell_w = w // grid_cols
        cell_h = h // grid_rows

        for cell_idx in range(grid_rows * grid_cols):
            row = cell_idx // grid_cols
            col = cell_idx % grid_cols
            x1 = col * cell_w
            y1 = row * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            if cell_idx in result.grid_cells:
                # Green overlay for selected cells
                overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle([(x1, y1), (x2, y2)], fill=self.GREEN_TRANSPARENT)
                overlay_draw.rectangle([(x1, y1), (x2, y2)], outline=self.GREEN_BORDER, width=3)
                annotated = Image.alpha_composite(annotated, overlay)
            else:
                # Slight dark overlay for unselected
                overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle([(x1, y1), (x2, y2)], fill=(0, 0, 0, 40))
                annotated = Image.alpha_composite(annotated, overlay)

        # Draw grid lines
        draw = ImageDraw.Draw(annotated)
        for row in range(grid_rows + 1):
            y = row * cell_h
            draw.line([(0, y), (w, y)], fill=self.WHITE, width=1)
        for col in range(grid_cols + 1):
            x = col * cell_w
            draw.line([(x, 0), (x, h)], fill=self.WHITE, width=1)

        # Cell labels
        for cell_idx in range(grid_rows * grid_cols):
            row = cell_idx // grid_cols
            col = cell_idx % grid_cols
            cx = col * cell_w + cell_w // 2
            cy = row * cell_h + cell_h // 2

            if cell_idx in result.grid_cells:
                self._draw_checkmark(draw, cx - 10, cy - 10, 20, self.GREEN_BORDER)
            else:
                self._draw_xmark(draw, cx - 8, cy - 8, 16, (180, 180, 180, 150))

        # Question overlay
        self._draw_question_bar(draw, question, w)

        out_path = os.path.join(self.output_dir, "solution_grid.png")
        annotated.save(out_path, "PNG")
        print(f"\n  [Annotator] Saved: {out_path}")
        return out_path

    # ── Helper drawing methods ────────────────────────────────────────

    def _draw_label(
        self,
        draw: ImageDraw.Draw,
        x: int, y: int,
        text: str,
        bg_color: tuple = (60, 60, 60),
        text_color: tuple = (255, 255, 255),
        font=None,
    ):
        """Draw a text label with background."""
        if font is None:
            font = self.font_small
        bbox = draw.textbbox((x, y), text, font=font)
        padding = 3
        draw.rectangle(
            [(bbox[0] - padding, bbox[1] - padding),
             (bbox[2] + padding, bbox[3] + padding)],
            fill=(*bg_color[:3], 200) if len(bg_color) == 3 else bg_color,
        )
        draw.text((x, y), text, fill=text_color, font=font)

    def _draw_question_bar(self, draw: ImageDraw.Draw, question: str, width: int):
        """Draw a question bar at the top of the image."""
        bar_height = 40
        draw.rectangle([(0, 0), (width, bar_height)], fill=(0, 0, 0, 180))
        wrapped = self._wrap_text(question, width - 20, self.font_medium)
        y = 5
        for line in wrapped[:2]:
            draw.text((10, y), line, fill=self.YELLOW, font=self.font_medium)
            y += 18

    def _draw_checkmark(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: tuple):
        """Draw a checkmark."""
        points = [
            (x, y + size * 0.5),
            (x + size * 0.35, y + size * 0.85),
            (x + size, y + size * 0.15),
        ]
        draw.line(points[:2], fill=color, width=3)
        draw.line(points[1:], fill=color, width=3)

    def _draw_xmark(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: tuple):
        """Draw an X mark."""
        draw.line([(x, y), (x + size, y + size)], fill=color, width=2)
        draw.line([(x + size, y), (x, y + size)], fill=color, width=2)

    def _draw_arrow(
        self,
        draw: ImageDraw.Draw,
        x1: int, y1: int,
        x2: int, y2: int,
        color: tuple = (255, 220, 0),
        width: int = 3,
    ):
        """Draw an arrow from (x1,y1) to (x2,y2)."""
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

        # Arrowhead
        angle = math.atan2(y2 - y1, x2 - x1)
        head_len = 15
        head_angle = math.pi / 6  # 30 degrees

        left_x = x2 - head_len * math.cos(angle - head_angle)
        left_y = y2 - head_len * math.sin(angle - head_angle)
        right_x = x2 - head_len * math.cos(angle + head_angle)
        right_y = y2 - head_len * math.sin(angle + head_angle)

        draw.polygon(
            [(x2, y2), (int(left_x), int(left_y)), (int(right_x), int(right_y))],
            fill=color,
        )

    def _wrap_text(self, text: str, max_width: int, font) -> List[str]:
        """Simple text wrapping."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test = f"{current_line} {word}".strip()
            try:
                bbox = font.getbbox(test)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(test) * 8

            if text_width <= max_width:
                current_line = test
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines if lines else [text]


# ══════════════════════════════════════════════════════════════════════
# MAIN SOLVER — Integrates everything
# ══════════════════════════════════════════════════════════════════════

class HCaptchaCLIPSolver:
    """
    Full hCaptcha solver integrating:
    - Challenge fetching (from MinimalHCaptchaImageDownloader)
    - CLIP AI analysis
    - Visual annotation
    - Solution submission
    """

    HC_API = "https://api2.hcaptcha.com"
    HC_ASSETS = "https://newassets.hcaptcha.com"

    def __init__(
        self,
        site_key: str,
        site_url: str,
        output_dir: str = "hcaptcha_solutions",
        debug: bool = False,
        clip_model: str = "ViT-L/14@336px",
        device: str = None,
    ):
        self.site_key = site_key
        self.site_url = site_url
        self.host = urlparse(site_url).hostname
        self.output_dir = output_dir
        self.version = None
        self.hsw_path = None
        self.debug = debug
        self._last_entity_placements: Dict[int, List[Dict]] = {}

        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize HTTP session
        self.session = requests.Session()
        ua = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        )
        self.session.headers.update({
            "User-Agent": ua,
            "Accept-Language": "en-US,en;q=0.9",
            "sec-ch-ua": '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
        })
        
        self.proxies = []
        proxy_file = "proxies.txt"
        if os.path.exists(proxy_file):
            with open(proxy_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and "@" in line:
                        # Format: user:pass@ip:port
                        self.proxies.append(f"http://{line}")
            print(f"[+] Loaded {len(self.proxies)} proxies")
        else:
            print(f"[!] No proxies.txt found, running without proxies")

        self._rotate_proxy()

        # Initialize CLIP solver
        self.solver = CLIPSolverEngine(device=device, model_name=clip_model)

        # Initialize annotator
        self.annotator = SolutionAnnotator(output_dir)

        print(f"[+] HCaptcha CLIP Solver initialized | {site_url}")

    def _dbg(self, msg):
        if self.debug:
            print(f"    [DBG] {msg}")

    def _decode_jwt(self, token):
        try:
            p = token.split(".")[1]
            p += "=" * (4 - len(p) % 4)
            return json.loads(base64.urlsafe_b64decode(p))
        except:
            return {}

    def _gen_widget_id(self):
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))

    def _gen_motion_data(self):
        now = int(time.time() * 1000)
        page_load = now - random.randint(5000, 15000)
        widget_show = now - random.randint(2000, 5000)
        wid = self._gen_widget_id()

        top_mm = []
        t = page_load + random.randint(500, 2000)
        x, y = random.randint(200, 600), random.randint(200, 400)
        for _ in range(random.randint(5, 15)):
            x = max(0, min(1920, x + random.randint(-60, 60)))
            y = max(0, min(1080, y + random.randint(-60, 60)))
            t += random.randint(15, 80)
            top_mm.append([x, y, t])

        mm = []
        t = widget_show + random.randint(200, 800)
        x, y = random.randint(10, 30), random.randint(10, 30)
        for _ in range(random.randint(3, 12)):
            x = max(0, min(300, x + random.randint(-5, 30)))
            y = max(0, min(75, y + random.randint(-5, 20)))
            t += random.randint(15, 60)
            mm.append([x, y, t])

        cx = random.randint(20, 35)
        cy = random.randint(25, 40)
        ct = (mm[-1][2] if mm else t) + random.randint(50, 200)
        md = [[cx, cy, ct]]
        mu = [[cx + random.randint(-1, 1), cy + random.randint(-1, 1), ct + random.randint(60, 150)]]

        return {
            "st": widget_show,
            "dct": widget_show,
            "mm": mm,
            "mm-mp": 0,
            "md": md,
            "md-mp": 0,
            "mu": mu,
            "mu-mp": 0,
            "v": 1,
            "topLevel": {
                "st": page_load,
                "sc": {
                    "availWidth": 1920, "availHeight": 1040,
                    "width": 1920, "height": 1080,
                    "colorDepth": 24, "pixelDepth": 24,
                    "availLeft": 0, "availTop": 0,
                },
                "nv": {
                    "vendorSub": "",
                    "productSub": "20030107",
                    "vendor": "Google Inc.",
                    "maxTouchPoints": 0,
                    "userActivation": {},
                    "doNotTrack": None,
                    "geolocation": {},
                    "connection": {},
                    "webkitTemporaryStorage": {},
                    "webkitPersistentStorage": {},
                    "hardwareConcurrency": 8,
                    "cookieEnabled": True,
                    "appCodeName": "Mozilla",
                    "appName": "Netscape",
                    "appVersion": "5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
                    "platform": "Win32",
                    "product": "Gecko",
                    "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
                    "language": "en-US",
                    "languages": ["en-US", "en"],
                    "onLine": True,
                    "webdriver": False,
                    "pdfViewerEnabled": True,
                    "deviceMemory": 8,
                    "plugins": [],
                },
                "dr": "",
                "inv": False,
                "exec": False,
                "wn": [[1366, 768, page_load + 100]],
                "wn-mp": 0,
                "xy": [[0, 0, page_load + 100]],
                "xy-mp": 0,
                "mm": top_mm,
                "mm-mp": 0,
            },
            "session": [],
            "widgetList": [wid],
            "widgetId": wid,
            "href": self.site_url,
            "prev": {
                "escaped": False,
                "passed": False,
                "expiredChallenge": False,
                "expiredResponse": False,
            },
        }

    # ── API interaction steps ─────────────────────────────────────────

    def _check_site_config(self):
        print("\n[1] Getting site config...")
        try:
            resp = self.session.get(
                f"{self.HC_API}/checksiteconfig",
                params={"v": "", "host": self.host, "sitekey": self.site_key, "sc": "1", "swa": "1"},
                headers={"Accept": "application/json", "Referer": self.site_url},
                timeout=15,
            )
            data = resp.json()
            jwt_req = data.get("c", {}).get("req", "")
            if jwt_req:
                l = self._decode_jwt(jwt_req).get("l", "")
                if l:
                    self.version = l.strip("/").split("/")[-1]
                    print(f"    Version: {self.version[:24]}...")
            print("    ✓ Config received")
            return data
        except Exception as e:
            print(f"    ✗ {e}")
            return None

    def _solve_hsw(self, jwt_req):
        if not self.hsw_path:
            return None
        try:
            subprocess.run(["node", "-v"], capture_output=True, timeout=5)
        except:
            print("    ✗ Node.js not found!")
            return None

        runner = self._build_runner(jwt_req)
        fd, path = tempfile.mkstemp(suffix=".js", prefix="runner_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(runner)
            t0 = time.time()
            r = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
            dt = time.time() - t0
            os.unlink(path)
            if r.returncode == 0 and r.stdout.strip():
                proof = r.stdout.strip()
                print(f"    ✓ Proof ({len(proof)} chars, {dt:.1f}s)")
                return proof
            print(f"    ✗ Proof failed")
            if r.stderr:
                print(f"    {r.stderr[:400]}")
            return None
        except Exception as e:
            print(f"    ✗ {e}")
            if os.path.exists(path):
                os.unlink(path)
            return None

    def _build_runner(self, jwt_token):
        """Build the Node.js runner script for HSW proof generation."""
        safe = jwt_token.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$").replace('"', '\\"')
        hsw = os.path.abspath(self.hsw_path).replace("\\", "/")
        ver = self.version or ""
        # Using the same runner from MinimalHCaptchaImageDownloader
        return (
            f'var _p = process, _B = Buffer, _r = require;\n'
            f'var _fs = _r("fs"), _crypto = _r("crypto"), _util = _r("util"), _url = _r("url");\n'
            f'var G = globalThis;\n'
            f'G.window = G; G.self = G; G.top = G; G.parent = G; G.frames = G;\n'
            f'G.navigator = {{ userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36", language: "en-US", languages: ["en-US","en"], platform: "Win32", hardwareConcurrency: 8, deviceMemory: 8, maxTouchPoints: 0, webdriver: false, vendor: "Google Inc.", appVersion: "5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36", appName: "Netscape", appCodeName: "Mozilla", product: "Gecko", productSub: "20030107", vendorSub: "", cookieEnabled: true, onLine: true, doNotTrack: null, pdfViewerEnabled: true, plugins: [], mimeTypes: [], userAgentData: {{ brands: [{{ brand: "Chromium", version: "130" }}, {{ brand: "Google Chrome", version: "130" }}], mobile: false, platform: "Windows" }}, getBattery: function() {{ return Promise.resolve({{ charging: true, chargingTime: 0, dischargingTime: Infinity, level: 1 }}); }}, getGamepads: function() {{ return []; }}, javaEnabled: function() {{ return false; }}, sendBeacon: function() {{ return true; }}, vibrate: function() {{ return true; }} }};\n'
            f'G.location = {{ href: "https://newassets.hcaptcha.com/captcha/v1/{ver}/static/hcaptcha.html", origin: "https://newassets.hcaptcha.com", hostname: "newassets.hcaptcha.com", host: "newassets.hcaptcha.com", protocol: "https:", pathname: "/captcha/v1/{ver}/static/hcaptcha.html", port: "", search: "", hash: "", assign: function() {{}}, reload: function() {{}}, replace: function() {{}}, toString: function() {{ return this.href; }} }};\n'
            f'G.screen = {{ width: 1920, height: 1080, availWidth: 1920, availHeight: 1040, colorDepth: 24, pixelDepth: 24, orientation: {{ type: "landscape-primary", angle: 0 }} }};\n'
            f'G.innerWidth = 1366; G.innerHeight = 768; G.outerWidth = 1366; G.outerHeight = 839; G.devicePixelRatio = 1;\n'
            f'G.screenX = 0; G.screenY = 0; G.screenLeft = 0; G.screenTop = 0;\n'
            f'G.pageXOffset = 0; G.pageYOffset = 0; G.scrollX = 0; G.scrollY = 0;\n'
            f'var perfStart = Date.now() - 5000;\n'
            f'G.performance = {{ now: function() {{ return Date.now() - perfStart; }}, timeOrigin: perfStart, timing: {{ navigationStart: perfStart, fetchStart: perfStart+1, domainLookupStart: perfStart+2, domainLookupEnd: perfStart+5, connectStart: perfStart+5, connectEnd: perfStart+50, requestStart: perfStart+51, responseStart: perfStart+120, responseEnd: perfStart+180, domLoading: perfStart+200, domInteractive: perfStart+800, domContentLoadedEventStart: perfStart+900, domContentLoadedEventEnd: perfStart+920, domComplete: perfStart+1200, loadEventStart: perfStart+1250, loadEventEnd: perfStart+1300 }}, navigation: {{ type: 0, redirectCount: 0 }}, getEntries: function() {{ return []; }}, getEntriesByType: function() {{ return []; }}, getEntriesByName: function() {{ return []; }}, mark: function() {{}}, measure: function() {{}} }};\n'
            f'function makeCtx2d() {{ return {{ fillRect:function(){{}}, clearRect:function(){{}}, fillText:function(){{}}, strokeText:function(){{}}, measureText:function(t){{ return {{width:t.length*8}}; }}, getImageData:function(x,y,w,h){{ return {{data:new Uint8ClampedArray(w*h*4)}}; }}, putImageData:function(){{}}, createImageData:function(w,h){{ return {{data:new Uint8ClampedArray(w*h*4)}}; }}, setTransform:function(){{}}, resetTransform:function(){{}}, drawImage:function(){{}}, save:function(){{}}, restore:function(){{}}, beginPath:function(){{}}, moveTo:function(){{}}, lineTo:function(){{}}, closePath:function(){{}}, stroke:function(){{}}, arc:function(){{}}, fill:function(){{}}, rect:function(){{}}, clip:function(){{}}, translate:function(){{}}, rotate:function(){{}}, scale:function(){{}}, transform:function(){{}}, createLinearGradient:function(){{ return {{addColorStop:function(){{}}}}; }}, createRadialGradient:function(){{ return {{addColorStop:function(){{}}}}; }}, createPattern:function(){{ return null; }}, globalCompositeOperation:"source-over", fillStyle:"#000000", strokeStyle:"#000000", lineWidth:1, font:"10px sans-serif", textBaseline:"alphabetic", textAlign:"start", globalAlpha:1, shadowBlur:0, shadowColor:"rgba(0,0,0,0)", shadowOffsetX:0, shadowOffsetY:0, canvas:{{width:300,height:150}} }}; }}\n'
            f'function makeWebGL() {{ return {{ getParameter:function(p){{ if(p===7937)return"WebKit";if(p===7936)return"WebKit WebGL";if(p===37446)return"ANGLE (Intel, Intel(R) UHD Graphics 630)";if(p===37445)return"Google Inc. (Intel)";if(p===7938)return"WebGL 1.0";if(p===35724)return"WebGL GLSL ES 1.0";return""; }}, getSupportedExtensions:function(){{ return []; }}, getExtension:function(n){{ if(n==="WEBGL_debug_renderer_info")return {{UNMASKED_VENDOR_WEBGL:37445,UNMASKED_RENDERER_WEBGL:37446}}; return null; }}, createBuffer:function(){{ return {{}}; }}, createProgram:function(){{ return {{}}; }}, createShader:function(){{ return {{}}; }}, shaderSource:function(){{}}, compileShader:function(){{}}, attachShader:function(){{}}, linkProgram:function(){{}}, useProgram:function(){{}}, getShaderParameter:function(){{ return true; }}, getProgramParameter:function(){{ return true; }}, bindBuffer:function(){{}}, bufferData:function(){{}}, enableVertexAttribArray:function(){{}}, vertexAttribPointer:function(){{}}, drawArrays:function(){{}}, viewport:function(){{}}, clearColor:function(){{}}, clear:function(){{}}, getAttribLocation:function(){{ return 0; }}, getUniformLocation:function(){{ return {{}}; }}, uniform1f:function(){{}}, uniform2f:function(){{}}, canvas:{{width:300,height:150}}, VERTEX_SHADER:35633, FRAGMENT_SHADER:35632, COMPILE_STATUS:35713, LINK_STATUS:35714, ARRAY_BUFFER:34962, STATIC_DRAW:35044, FLOAT:5126, TRIANGLES:4, COLOR_BUFFER_BIT:16384, DEPTH_BUFFER_BIT:256 }}; }}\n'
            f'function makeCanvas() {{ return {{ getContext:function(t){{ if(t==="2d")return makeCtx2d(); if(t==="webgl"||t==="webgl2"||t==="experimental-webgl")return makeWebGL(); return null; }}, toDataURL:function(){{ return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="; }}, toBlob:function(cb){{ if(cb)cb(null); }}, width:300, height:150, style:{{}}, setAttribute:function(){{}}, getAttribute:function(){{ return null; }}, addEventListener:function(){{}}, getBoundingClientRect:function(){{ return {{top:0,left:0,bottom:150,right:300,width:300,height:150}}; }} }}; }}\n'
            f'function makeEl() {{ return {{ style:{{}}, setAttribute:function(){{}}, getAttribute:function(){{ return null; }}, appendChild:function(){{}}, removeChild:function(){{}}, insertBefore:function(){{}}, addEventListener:function(){{}}, removeEventListener:function(){{}}, getElementsByTagName:function(){{ return []; }}, getElementsByClassName:function(){{ return []; }}, querySelector:function(){{ return null; }}, querySelectorAll:function(){{ return []; }}, getBoundingClientRect:function(){{ return {{top:0,left:0,bottom:0,right:0,width:0,height:0}}; }}, cloneNode:function(){{ return makeEl(); }}, innerHTML:"", textContent:"", id:"", className:"", classList:{{add:function(){{}},remove:function(){{}},contains:function(){{ return false; }},toggle:function(){{}}}}, dataset:{{}}, offsetWidth:0, offsetHeight:0, offsetLeft:0, offsetTop:0, clientWidth:0, clientHeight:0, scrollWidth:0, scrollHeight:0, childNodes:[], children:[], firstChild:null, lastChild:null, parentNode:null, parentElement:null, nextSibling:null, previousSibling:null, nodeName:"DIV", nodeType:1, ownerDocument:null, dispatchEvent:function(){{ return true; }}, focus:function(){{}}, blur:function(){{}}, click:function(){{}} }}; }}\n'
            f'G.document = {{ createElement:function(tag){{ if(tag==="canvas")return makeCanvas(); return makeEl(); }}, createElementNS:function(){{ return makeCanvas(); }}, createTextNode:function(){{ return {{textContent:""}}; }}, createDocumentFragment:function(){{ return {{appendChild:function(){{}},childNodes:[]}}; }}, querySelector:function(){{ return null; }}, querySelectorAll:function(){{ return []; }}, getElementById:function(){{ return null; }}, getElementsByTagName:function(){{ return []; }}, getElementsByClassName:function(){{ return []; }}, getElementsByName:function(){{ return []; }}, head:{{appendChild:function(){{}},removeChild:function(){{}},querySelector:function(){{ return null; }},childNodes:[]}}, body:{{appendChild:function(){{}},removeChild:function(){{}},style:{{}},getBoundingClientRect:function(){{ return {{top:0,left:0,bottom:768,right:1366,width:1366,height:768}}; }}}}, documentElement:{{style:{{}},getAttribute:function(){{ return null; }},clientWidth:1366,clientHeight:768}}, cookie:"", readyState:"complete", title:"", domain:"newassets.hcaptcha.com", referrer:"", URL:"https://newassets.hcaptcha.com/captcha/v1/{ver}/static/hcaptcha.html", hasFocus:function(){{ return true; }}, hidden:false, visibilityState:"visible", addEventListener:function(){{}}, removeEventListener:function(){{}}, createEvent:function(){{ return {{initEvent:function(){{}}}}; }}, dispatchEvent:function(){{ return true; }}, adoptNode:function(n){{ return n; }}, importNode:function(n){{ return n; }}, createComment:function(){{ return {{nodeName:"#comment"}}; }}, implementation:{{hasFeature:function(){{ return true; }},createHTMLDocument:function(){{ return G.document; }}}}, characterSet:"UTF-8", contentType:"text/html", compatMode:"CSS1Compat", doctype:{{name:"html"}} }};\n'
            f'if (_crypto.webcrypto) {{ G.crypto = _crypto.webcrypto; }} else {{ G.crypto = {{ getRandomValues:function(arr){{ _crypto.randomFillSync(arr); return arr; }}, subtle:{{ digest:function(alg,data){{ var name=typeof alg==="string"?alg:alg.name; var h=_crypto.createHash(name.replace("-","").toLowerCase()); h.update(new Uint8Array(data)); var r=h.digest(); return Promise.resolve(r.buffer.slice(r.byteOffset,r.byteOffset+r.byteLength)); }}, importKey:function(){{ return Promise.resolve({{type:"secret"}}); }}, exportKey:function(){{ return Promise.resolve(new ArrayBuffer(0)); }}, sign:function(){{ return Promise.resolve(new ArrayBuffer(32)); }}, verify:function(){{ return Promise.resolve(true); }}, encrypt:function(){{ return Promise.resolve(new ArrayBuffer(0)); }}, decrypt:function(){{ return Promise.resolve(new ArrayBuffer(0)); }}, deriveBits:function(){{ return Promise.resolve(new ArrayBuffer(32)); }}, deriveKey:function(){{ return Promise.resolve({{type:"secret"}}); }} }}, randomUUID:function(){{ return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g,function(c){{ var r=Math.random()*16|0; return (c==="x"?r:(r&0x3|0x8)).toString(16); }}); }} }}; }}\n'
            f'G.atob = function(s){{ return _B.from(s,"base64").toString("binary"); }};\n'
            f'G.btoa = function(s){{ return _B.from(s,"binary").toString("base64"); }};\n'
            f'G.TextEncoder = _util.TextEncoder; G.TextDecoder = _util.TextDecoder;\n'
            f'var stor = {{}};\n'
            f'G.localStorage = {{ getItem:function(k){{ return stor[k]||null; }}, setItem:function(k,v){{ stor[k]=String(v); }}, removeItem:function(k){{ delete stor[k]; }}, clear:function(){{ stor={{}}; }}, get length(){{ return Object.keys(stor).length; }}, key:function(i){{ return Object.keys(stor)[i]||null; }} }};\n'
            f'G.sessionStorage = G.localStorage;\n'
            f'G.history = {{ length:2, state:null, pushState:function(){{}}, replaceState:function(){{}}, back:function(){{}}, forward:function(){{}}, go:function(){{}} }};\n'
            f'G.Image = function(){{ this.src=""; this.width=0; this.height=0; this.onload=null; this.onerror=null; }};\n'
            f'G.Audio = function(){{ this.src=""; this.play=function(){{}}; this.pause=function(){{}}; }};\n'
            f'G.Blob = function(p,o){{ this.size=0; this.type=(o||{{}}).type||""; }};\n'
            f'G.File = function(){{ this.name=""; this.size=0; }};\n'
            f'G.FileReader = function(){{ this.readAsDataURL=function(){{}}; this.readAsText=function(){{}}; }};\n'
            f'G.URL = _url.URL;\n'
            f'G.URL.createObjectURL = function(){{ return "blob:https://newassets.hcaptcha.com/"+Math.random().toString(36).slice(2); }};\n'
            f'G.URL.revokeObjectURL = function(){{}};\n'
            f'G.Worker = function(){{ this.postMessage=function(){{}}; this.terminate=function(){{}}; this.addEventListener=function(){{}}; }};\n'
            f'G.MessageChannel = function(){{ this.port1={{postMessage:function(){{}},addEventListener:function(){{}},close:function(){{}}}}; this.port2={{postMessage:function(){{}},addEventListener:function(){{}},close:function(){{}}}}; }};\n'
            f'G.BroadcastChannel = function(){{ this.postMessage=function(){{}}; this.close=function(){{}}; this.addEventListener=function(){{}}; }};\n'
            f'G.requestAnimationFrame = function(cb){{ return setTimeout(cb,16); }};\n'
            f'G.cancelAnimationFrame = function(id){{ clearTimeout(id); }};\n'
            f'G.requestIdleCallback = function(cb){{ return setTimeout(function(){{ cb({{didTimeout:false,timeRemaining:function(){{ return 50; }}}}); }},1); }};\n'
            f'G.MutationObserver = function(){{ this.observe=function(){{}}; this.disconnect=function(){{}}; this.takeRecords=function(){{ return []; }}; }};\n'
            f'G.IntersectionObserver = function(){{ this.observe=function(){{}}; this.unobserve=function(){{}}; this.disconnect=function(){{}}; }};\n'
            f'G.ResizeObserver = function(){{ this.observe=function(){{}}; this.unobserve=function(){{}}; this.disconnect=function(){{}}; }};\n'
            f'G.PerformanceObserver = function(){{ this.observe=function(){{}}; this.disconnect=function(){{}}; }};\n'
            f'G.getComputedStyle = function(){{ return new Proxy({{}}, {{ get: function(){{ return ""; }} }}); }};\n'
            f'G.matchMedia = function(){{ return {{ matches:false, media:"", addEventListener:function(){{}}, removeEventListener:function(){{}}, addListener:function(){{}}, removeListener:function(){{}} }}; }};\n'
            f'G.postMessage = function(){{}}; G.addEventListener = function(){{}}; G.removeEventListener = function(){{}}; G.dispatchEvent = function(){{ return true; }};\n'
            f'G.open = function(){{ return null; }}; G.close = function(){{}}; G.focus = function(){{}}; G.blur = function(){{}};\n'
            f'G.fetch = function(){{ return Promise.resolve({{ ok:true, status:200, json:function(){{ return Promise.resolve({{}}); }}, text:function(){{ return Promise.resolve(""); }}, headers:{{ get:function(){{ return null; }} }} }}); }};\n'
            f'G.XMLHttpRequest = function(){{ this.open=function(){{}}; this.send=function(){{}}; this.setRequestHeader=function(){{}}; this.addEventListener=function(){{}}; this.status=200; this.readyState=4; this.responseText=""; }};\n'
            f'G.AbortController = function(){{ this.signal={{aborted:false,addEventListener:function(){{}}}}; this.abort=function(){{}}; }};\n'
            f'G.Event = function(t){{ this.type=t; this.preventDefault=function(){{}}; this.stopPropagation=function(){{}}; }};\n'
            f'G.CustomEvent = function(t,p){{ this.type=t; this.detail=(p||{{}}).detail; }};\n'
            f'G.ErrorEvent = function(){{}}; G.PromiseRejectionEvent = function(){{}};\n'
            f'G.DOMParser = function(){{ this.parseFromString=function(){{ return G.document; }}; }};\n'
            f'G.Notification = {{ permission:"default", requestPermission:function(){{ return Promise.resolve("default"); }} }};\n'
            f'G.MediaStream = function(){{}};\n'
            f'G.RTCPeerConnection = function(){{ this.createDataChannel=function(){{ return {{}}; }}; this.createOffer=function(){{ return Promise.resolve({{}}); }}; this.setLocalDescription=function(){{ return Promise.resolve(); }}; this.close=function(){{}}; }};\n'
            f'G.AudioContext = function(){{ this.createAnalyser=function(){{ return {{connect:function(){{}},frequencyBinCount:128,getFloatFrequencyData:function(a){{a.fill(-100);}}}}; }}; this.createOscillator=function(){{ return {{connect:function(){{}},start:function(){{}},stop:function(){{}},type:"sine",frequency:{{value:440}}}}; }}; this.createDynamicsCompressor=function(){{ return {{connect:function(){{}},threshold:{{value:-24}},knee:{{value:30}},ratio:{{value:12}},attack:{{value:0.003}},release:{{value:0.25}}}}; }}; this.destination={{}}; this.sampleRate=44100; this.close=function(){{ return Promise.resolve(); }}; }};\n'
            f'G.webkitAudioContext = G.AudioContext;\n'
            f'G.OfflineAudioContext = function(){{ this.startRendering=function(){{ return Promise.resolve({{getChannelData:function(){{ return new Float32Array(44100); }}}}); }}; this.createDynamicsCompressor=function(){{ return {{connect:function(){{}}}}; }}; this.destination={{}}; }};\n'
            f'G.chrome = {{ runtime:{{id:undefined}}, loadTimes:function(){{ return {{}}; }}, csi:function(){{ return {{}}; }} }};\n'
            f'G.Intl = G.Intl || {{}};\n'
            f'G.SharedArrayBuffer = G.SharedArrayBuffer || ArrayBuffer;\n'
            f'G.structuredClone = G.structuredClone || function(o){{ return JSON.parse(JSON.stringify(o)); }};\n'
            f'G.queueMicrotask = G.queueMicrotask || function(fn){{ Promise.resolve().then(fn); }};\n'
            f'G.reportError = function(e){{}}; G.isSecureContext = true; G.origin = "https://newassets.hcaptcha.com"; G.crossOriginIsolated = false;\n'
            f'try {{ delete G.process; }} catch(e) {{ Object.defineProperty(G, "process", {{ value: undefined, configurable: true, writable: true }}); }}\n'
            f'try {{ delete G.Buffer; }} catch(e) {{ Object.defineProperty(G, "Buffer", {{ value: undefined, configurable: true, writable: true }}); }}\n'
            f'try {{ delete G.global; }} catch(e) {{ Object.defineProperty(G, "global", {{ value: undefined, configurable: true, writable: true }}); }}\n'
            f'try {{ delete G.setImmediate; }} catch(e) {{}}\n'
            f'try {{ delete G.clearImmediate; }} catch(e) {{}}\n'
            f'try {{ delete G.GLOBAL; }} catch(e) {{}}\n'
            f'try {{ delete G.root; }} catch(e) {{}}\n'
            f'var hswCode = _fs.readFileSync("{hsw}", "utf8");\n'
            f'(new Function(hswCode))();\n'
            f'var hswFunc = G.hsw || G.h || G.H;\n'
            f'if (!hswFunc) {{\n'
            f'  var keys = Object.getOwnPropertyNames(G);\n'
            f'  var builtins = new Set(["window","self","top","parent","frames","navigator","location","document","screen","performance","crypto","localStorage","sessionStorage","history","chrome","Intl","Image","Audio","Blob","File","FileReader","URL","Worker","MessageChannel","BroadcastChannel","MutationObserver","IntersectionObserver","ResizeObserver","PerformanceObserver","Event","CustomEvent","ErrorEvent","PromiseRejectionEvent","DOMParser","Notification","MediaStream","RTCPeerConnection","AudioContext","webkitAudioContext","OfflineAudioContext","XMLHttpRequest","AbortController","TextEncoder","TextDecoder","atob","btoa","fetch","postMessage","addEventListener","removeEventListener","dispatchEvent","requestAnimationFrame","cancelAnimationFrame","requestIdleCallback","getComputedStyle","matchMedia","open","close","focus","blur","setTimeout","clearTimeout","setInterval","clearInterval","queueMicrotask","structuredClone","Array","ArrayBuffer","BigInt","BigInt64Array","BigUint64Array","Boolean","DataView","Date","Error","EvalError","Float32Array","Float64Array","Function","Infinity","Int16Array","Int32Array","Int8Array","JSON","Map","Math","NaN","Number","Object","Promise","Proxy","RangeError","ReferenceError","Reflect","RegExp","Set","SharedArrayBuffer","String","Symbol","SyntaxError","TypeError","URIError","Uint16Array","Uint32Array","Uint8Array","Uint8ClampedArray","WeakMap","WeakRef","WeakSet","WebAssembly","undefined","decodeURI","decodeURIComponent","encodeURI","encodeURIComponent","escape","unescape","eval","globalThis","isFinite","isNaN","parseFloat","parseInt","console","reportError","isSecureContext","origin","crossOriginIsolated","innerWidth","innerHeight","outerWidth","outerHeight","devicePixelRatio","screenX","screenY","screenLeft","screenTop","pageXOffset","pageYOffset","scrollX","scrollY","AggregateError","FinalizationRegistry"]);\n'
            f'  for (var i=0; i<keys.length; i++) {{ var k = keys[i]; if (!builtins.has(k) && typeof G[k] === "function" && k.length <= 4) {{ hswFunc = G[k]; break; }} }}\n'
            f'}}\n'
            f'if (!hswFunc) {{ _p.stderr.write("Cannot find hsw function\\n"); _p.exit(1); }}\n'
            f'var jwt = "{safe}";\n'
            f'Promise.resolve(hswFunc(jwt)).then(function(result) {{\n'
            f'  if (result && typeof result === "string") {{ _p.stdout.write(result); _p.exit(0); }}\n'
            f'  else {{ _p.stderr.write("Invalid result: " + typeof result + "\\n"); _p.exit(1); }}\n'
            f'}}).catch(function(err) {{ _p.stderr.write("Error: " + err.message + "\\n" + (err.stack||"") + "\\n"); _p.exit(1); }});\n'
        )

    def _get_challenge(self, c_data, proof):
        motion = self._gen_motion_data()
        form = {
            "v": self.version or "",
            "sitekey": self.site_key,
            "host": self.host,
            "hl": "en",
            "motionData": json.dumps(motion, separators=(",", ":")),
            "n": proof,
            "c": json.dumps(c_data, separators=(",", ":")),
        }

        resp = self.session.post(
            f"{self.HC_API}/getcaptcha/{self.site_key}",
            data=form,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "Origin": self.HC_ASSETS,
                "Referer": f"{self.HC_ASSETS}/captcha/v1/{self.version}/static/hcaptcha.html",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-site",
            },
            timeout=30,
        )
        return resp.json()

    # ── Challenge type detection ──────────────────────────────────────

    def _detect_challenge_type(self, challenge: Dict) -> ChallengeType:
        request_type = challenge.get("request_type", "")
        tasklist = challenge.get("tasklist", [])
        question = challenge.get("requester_question", {})
        if isinstance(question, dict):
            question = question.get("en", "")
        question_lower = question.lower()

        # Check for entities in tasks — definitive drag-drop
        if tasklist and any(t.get("entities") for t in tasklist):
            return ChallengeType.DRAG_DROP

        if request_type == "image_drag_drop":
            return ChallengeType.DRAG_DROP
        elif request_type == "image_label_binary":
            return ChallengeType.IMAGE_CLASSIFICATION
        elif request_type == "image_label_area_select":
            if "drag" in question_lower:
                return ChallengeType.DRAG_DROP
            return ChallengeType.BOUNDING_BOX
        elif request_type == "image_label_multiple_choice":
            return ChallengeType.MULTI_CHOICE
        elif request_type == "image_label_drag":
            return ChallengeType.DRAG_DROP
        elif request_type == "image_label_point":
            return ChallengeType.POINT_CLICK

        if "drag" in question_lower and ("drop" in question_lower or "to" in question_lower or "puzzle" in question_lower):
            return ChallengeType.DRAG_DROP
        if "click on" in question_lower or "click the" in question_lower or "point" in question_lower:
            if len(tasklist) <= 2:
                return ChallengeType.POINT_CLICK
        if "select" in question_lower or "click" in question_lower or "choose" in question_lower:
            return ChallengeType.IMAGE_CLASSIFICATION
        if "which" in question_lower:
            return ChallengeType.MULTI_CHOICE

        if len(tasklist) > 2:
            return ChallengeType.IMAGE_CLASSIFICATION
        elif len(tasklist) == 1:
            return ChallengeType.POINT_CLICK

        return ChallengeType.UNKNOWN


    # ── Image loading ─────────────────────────────────────────────────

    def _load_image_from_url(self, url: str) -> Optional[Image.Image]:
        """Download and load an image from URL as PIL Image."""
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 100:
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                return img
        except Exception as e:
            self._dbg(f"Failed to load image: {url[:60]}... - {e}")
        return None

    def _load_images_from_challenge(
        self, challenge: Dict
    ) -> Tuple[List[Image.Image], List[str], List[str]]:
        """
        Load all task images from the challenge.
        Returns (images, task_keys, image_urls).
        """
        tasklist = challenge.get("tasklist", [])
        images = []
        task_keys = []
        urls = []

        for i, task in enumerate(tasklist):
            url = task.get("datapoint_uri", "")
            key = task.get("task_key", f"task_{i}")

            if url:
                img = self._load_image_from_url(url)
                if img:
                    images.append(img)
                    task_keys.append(key)
                    urls.append(url)
                    print(f"    ✓ Loaded task {i}: {img.size[0]}x{img.size[1]}")
                else:
                    print(f"    ✗ Failed task {i}")
            else:
                print(f"    ✗ No URL for task {i}")

        return images, task_keys, urls

    def _load_example_images(self, challenge: Dict) -> List[Image.Image]:
        """Load example/reference images from the challenge."""
        examples = challenge.get("requester_question_example", [])
        if isinstance(examples, str):
            examples = [examples]

        images = []
        for url in examples:
            if isinstance(url, str) and url.startswith("http"):
                img = self._load_image_from_url(url)
                if img:
                    images.append(img)
                    print(f"    ✓ Loaded example: {img.size[0]}x{img.size[1]}")

        return images
        
        
    def _rotate_proxy(self):
        """Pick a random proxy and apply it to the session."""
        if not self.proxies:
            return
        proxy = random.choice(self.proxies)
        self.session.proxies = {
            "http": proxy,
            "https": proxy,
        }
        # Extract IP for logging
        ip = proxy.split("@")[-1] if "@" in proxy else proxy
        print(f"[+] Using proxy: {ip}")

    # ── Solving pipeline ──────────────────────────────────────────────

    def _solve_challenge(self, challenge: Dict) -> Optional[SolutionResult]:
        """
        Main solving pipeline — detects type and dispatches to appropriate solver.
        """
        challenge_type = self._detect_challenge_type(challenge)
        question = challenge.get("requester_question", {})
        if isinstance(question, dict):
            question = question.get("en", "")

        print(f"\n{'─'*60}")
        print(f"  Challenge Type: {challenge_type.value}")
        print(f"  Question: {question}")
        print(f"  Request Type: {challenge.get('request_type', 'N/A')}")
        print(f"  Tasks: {len(challenge.get('tasklist', []))}")
        print(f"{'─'*60}")

        # Parse the target from the question
        target, hint = self.solver.parse_question(question)
        print(f"  Parsed target: '{target}' (hint: {hint})")

        # Override challenge type based on hint
        if hint == "drag":
            challenge_type = ChallengeType.DRAG_DROP
        elif hint == "point":
            challenge_type = ChallengeType.POINT_CLICK

        # Load images
        print("\n[5] Loading challenge images...")
        images, task_keys, urls = self._load_images_from_challenge(challenge)
        example_images = self._load_example_images(challenge)

        if not images:
            print("    ✗ No images loaded!")
            return None

        # Dispatch to solver
        solution = None

        if challenge_type == ChallengeType.IMAGE_CLASSIFICATION:
            solution = self._solve_classification(
                challenge, images, task_keys, target, question
            )

        elif challenge_type == ChallengeType.MULTI_CHOICE:
            solution = self._solve_multi_choice(
                challenge, images, task_keys, target, question, example_images
            )

        elif challenge_type == ChallengeType.POINT_CLICK:
            solution = self._solve_point_click_challenge(
                challenge, images, task_keys, target, question
            )

        elif challenge_type == ChallengeType.BOUNDING_BOX:
            solution = self._solve_bounding_box_challenge(
                challenge, images, task_keys, target, question
            )

        elif challenge_type == ChallengeType.DRAG_DROP:
            solution = self._solve_drag_drop_challenge(
                challenge, images, task_keys, target, question, example_images
            )

        elif challenge_type == ChallengeType.GRID_CLASSIFICATION:
            solution = self._solve_grid_challenge(
                challenge, images, task_keys, target, question
            )

        else:
            # Unknown type — try classification as fallback
            print(f"  [!] Unknown type, falling back to classification")
            solution = self._solve_classification(
                challenge, images, task_keys, target, question
            )

        if solution:
            solution.challenge_type = challenge_type
            solution.question = question

        return solution

    def _solve_classification(
        self,
        challenge: Dict,
        images: List[Image.Image],
        task_keys: List[str],
        target: str,
        question: str,
    ) -> SolutionResult:
        """Solve image classification challenge."""
        print(f"\n[6] CLIP Classification Solver")

        results = self.solver.solve_image_classification(
            images, task_keys, target
        )

        # Build answer payload
        answers = {}
        for result in results:
            answers[result.task_key] = str(result.selected).lower()

        # Annotate
        annotated_path = self.annotator.annotate_classification(
            images, results, question, ChallengeType.IMAGE_CLASSIFICATION
        )

        solution = SolutionResult(
            challenge_type=ChallengeType.IMAGE_CLASSIFICATION,
            question=question,
            tasks=results,
            overall_confidence=np.mean([r.confidence for r in results]),
            solved=True,
            annotated_image_path=annotated_path,
            answer_payload=answers,
        )

        self._save_solution_data(solution, challenge)
        return solution

    def _solve_multi_choice(
        self,
        challenge: Dict,
        images: List[Image.Image],
        task_keys: List[str],
        target: str,
        question: str,
        example_images: List[Image.Image],
    ) -> SolutionResult:
        """
        Solve multi-choice: find which image best matches the target/example.
        """
        print(f"\n[6] CLIP Multi-Choice Solver")

        if example_images:
            # Compare each candidate to the example image
            example_features = self.solver.encode_images_batch(example_images)
            example_mean = F.normalize(example_features.mean(dim=0, keepdim=True), dim=-1)

            candidate_features = self.solver.encode_images_batch(images)
            similarities = (candidate_features @ example_mean.T).squeeze(-1).cpu().numpy()

            best_idx = int(np.argmax(similarities))

            results = []
            for i in range(len(images)):
                r = TaskResult(
                    task_index=i,
                    task_key=task_keys[i],
                    selected=(i == best_idx),
                    confidence=float(similarities[i]),
                    similarity_score=float(similarities[i]),
                    label=target,
                )
                results.append(r)
                status = "✓ BEST" if i == best_idx else "✗ skip"
                print(f"    [{i:2d}] {status}  sim={similarities[i]:.4f}")
        else:
            # No example — use text prompts
            results = self.solver.solve_image_classification(
                images, task_keys, target, threshold=0.0  # Select only the best
            )
            # Override: select only the top scoring one
            scores = [r.similarity_score for r in results]
            best_idx = int(np.argmax(scores))
            for i, r in enumerate(results):
                r.selected = (i == best_idx)

        answers = {}
        for r in results:
            answers[r.task_key] = str(r.selected).lower()

        annotated_path = self.annotator.annotate_classification(
            images, results, question, ChallengeType.MULTI_CHOICE
        )

        solution = SolutionResult(
            challenge_type=ChallengeType.MULTI_CHOICE,
            question=question,
            tasks=results,
            overall_confidence=max(r.confidence for r in results),
            solved=True,
            annotated_image_path=annotated_path,
            answer_payload=answers,
        )

        self._save_solution_data(solution, challenge)
        return solution

    def _solve_point_click_challenge(
        self,
        challenge: Dict,
        images: List[Image.Image],
        task_keys: List[str],
        target: str,
        question: str,
    ) -> SolutionResult:
        """Solve point-click challenge."""
        print(f"\n[6] CLIP Point-Click Solver")

        results = []
        annotated_paths = []

        for i, (img, key) in enumerate(zip(images, task_keys)):
            print(f"\n  --- Task {i} ---")
            result = self.solver.solve_point_click(img, key, target)
            result.task_index = i
            results.append(result)

            # Annotate each image
            path = self.annotator.annotate_point_click(img, result, question)
            if i > 0:
                # Rename for multiple tasks
                new_path = path.replace(".png", f"_task{i}.png")
                if os.path.exists(path):
                    os.rename(path, new_path)
                    path = new_path
            annotated_paths.append(path)

        # Build answer payload with click coordinates
        answers = {}
        for result in results:
            if result.click_point:
                # hCaptcha expects coordinates as a dict
                answers[result.task_key] = {
                    "x": result.click_point[0],
                    "y": result.click_point[1],
                }

        solution = SolutionResult(
            challenge_type=ChallengeType.POINT_CLICK,
            question=question,
            tasks=results,
            overall_confidence=np.mean([r.confidence for r in results]),
            solved=True,
            annotated_image_path=annotated_paths[0] if annotated_paths else "",
            answer_payload=answers,
        )

        self._save_solution_data(solution, challenge)
        return solution

    def _solve_bounding_box_challenge(
        self,
        challenge: Dict,
        images: List[Image.Image],
        task_keys: List[str],
        target: str,
        question: str,
    ) -> SolutionResult:
        """Solve bounding box challenge (similar to point-click)."""
        print(f"\n[6] CLIP Bounding Box Solver")

        # Same approach as point-click but with denser sampling
        results = []
        
        for i, (img, key) in enumerate(zip(images, task_keys)):
            print(f"\n  --- Task {i} ---")
            result = self.solver.solve_bounding_box(img, key, target, grid_divisions=12)
            result.task_index = i
            results.append(result)

            path = self.annotator.annotate_point_click(img, result, question)
            if i > 0:
                new_path = path.replace(".png", f"_bbox_task{i}.png")
                if os.path.exists(path):
                    os.rename(path, new_path)

        answers = {}
        for result in results:
            if result.click_point:
                answers[result.task_key] = {
                    "x": result.click_point[0],
                    "y": result.click_point[1],
                }

        solution = SolutionResult(
            challenge_type=ChallengeType.BOUNDING_BOX,
            question=question,
            tasks=results,
            overall_confidence=np.mean([r.confidence for r in results]),
            solved=True,
            annotated_image_path=os.path.join(self.output_dir, "solution_click.png"),
            answer_payload=answers,
        )

        self._save_solution_data(solution, challenge)
        return solution


    def _annotate_entity_drag(
        self,
        background: Image.Image,
        entity_images: List[Image.Image],
        placements: List[Dict],
        question: str,
        task_index: int,
    ) -> str:
        """
        Create annotated image showing entity drag-drop solution.
        - Green rectangles at drop targets
        - Red circles at entity start positions  
        - Yellow arrows from start to drop
        - Entity images overlaid at drop positions
        """
        annotated = background.copy().convert("RGBA")
        draw = ImageDraw.Draw(annotated)

        for pidx, placement in enumerate(placements):
            sx, sy = placement["start_x"], placement["start_y"]
            dx, dy = placement["drop_x"], placement["drop_y"]
            pw, ph = placement["size"]
            score = placement["score"]
            half_w, half_h = pw // 2, ph // 2

            # Green target zone at drop position
            target_overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
            target_draw = ImageDraw.Draw(target_overlay)
            target_draw.rectangle(
                [(dx - half_w, dy - half_h), (dx + half_w, dy + half_h)],
                fill=(0, 200, 0, 80),
                outline=(0, 255, 0),
                width=3,
            )
            annotated = Image.alpha_composite(annotated, target_overlay)
            draw = ImageDraw.Draw(annotated)

            # Overlay the entity image at drop position (semi-transparent)
            if pidx < len(entity_images):
                piece = entity_images[pidx].copy().convert("RGBA")
                piece = piece.resize((pw, ph), Image.LANCZOS)
                # Make semi-transparent
                alpha = piece.split()[-1] if piece.mode == "RGBA" else Image.new("L", piece.size, 255)
                alpha = alpha.point(lambda p: int(p * 0.7))
                piece.putalpha(alpha)
                paste_x = max(0, dx - half_w)
                paste_y = max(0, dy - half_h)
                annotated.paste(piece, (paste_x, paste_y), piece)
                draw = ImageDraw.Draw(annotated)

            # Red circle at start position
            r = 12
            draw.ellipse(
                [(sx - r, sy - r), (sx + r, sy + r)],
                outline=(255, 40, 40),
                width=3,
            )
            draw.ellipse(
                [(sx - 4, sy - 4), (sx + 4, sy + 4)],
                fill=(255, 40, 40),
            )

            # Yellow arrow from start to drop
            self.annotator._draw_arrow(draw, sx, sy, dx, dy, color=(255, 220, 0), width=3)

            # Green dot at drop center
            draw.ellipse(
                [(dx - 6, dy - 6), (dx + 6, dy + 6)],
                fill=(0, 200, 0),
                outline=(0, 255, 0),
                width=2,
            )

            # Labels
            self.annotator._draw_label(
                draw, sx + 15, sy - 15,
                f"START #{pidx}",
                bg_color=(220, 40, 40),
                font=self.annotator.font_small,
            )
            self.annotator._draw_label(
                draw, dx + half_w + 5, dy - 10,
                f"DROP #{pidx} ({dx},{dy})",
                bg_color=(0, 160, 0),
                font=self.annotator.font_small,
            )
            self.annotator._draw_label(
                draw, dx + half_w + 5, dy + 10,
                f"score: {score:.3f}",
                bg_color=(60, 60, 60),
                font=self.annotator.font_tiny,
            )

        # Question bar
        self.annotator._draw_question_bar(draw, question, annotated.size[0])

        out_path = os.path.join(self.output_dir, f"solution_drag_entity_{task_index}.png")
        annotated.save(out_path, "PNG")
        print(f"\n  [Annotator] Entity drag saved: {out_path}")
        return out_path

    def _solve_drag_drop_challenge(
        self,
        challenge: Dict,
        images: List[Image.Image],
        task_keys: List[str],
        target: str,
        question: str,
        example_images: List[Image.Image],
    ) -> SolutionResult:
        """Solve drag-and-drop — pure CLIP: paste piece everywhere in grid area, pick where it looks most complete."""
        print(f"\n[6] CLIP Drag-Drop Solver (Pure AI)")

        results = []
        tasklist = challenge.get("tasklist", [])
        is_normalized = challenge.get("normalized", False)

        good_features = self.solver.encode_text([
            "a complete natural image with nothing missing",
            "a perfectly assembled image",
            "a seamless photo with all parts in place",
            "a complete picture with no gaps",
            "an image that looks whole and correct",
        ])
        bad_features = self.solver.encode_text([
            "an image with a piece in the wrong place",
            "a broken image with misaligned parts",
            "a picture with something placed incorrectly",
            "an image with an obvious error",
            "a photo with a piece that does not belong",
        ])
        good_mean = F.normalize(good_features.mean(dim=0, keepdim=True), dim=-1)
        bad_mean = F.normalize(bad_features.mean(dim=0, keepdim=True), dim=-1)

        for i, (img, key) in enumerate(zip(images, task_keys)):
            print(f"\n  --- Task {i} ---")

            task = tasklist[i] if i < len(tasklist) else {}
            entities = task.get("entities", [])

            if not entities:
                result = self.solver.solve_point_click(img, key, target)
                result.task_index = i
                results.append(result)
                continue

            w, h = img.size
            entity_images_list = []
            entity_data = []

            for eidx, entity in enumerate(entities):
                entity_uri = entity.get("entity_uri", "")
                entity_id = entity.get("entity_id", "")
                start_coords = entity.get("coords", [0, 0])
                piece_size = entity.get("size", [66, 66])

                piece_img = self._load_image_from_url(entity_uri) if entity_uri else None
                if piece_img:
                    entity_images_list.append(piece_img)
                    entity_data.append({
                        "id": entity_id,
                        "image": piece_img,
                        "start_coords": start_coords,
                        "size": piece_size,
                        "index": eidx,
                    })
                    print(f"    Entity {eidx}: start=({start_coords[0]},{start_coords[1]}) size={piece_size}")

            if not entity_images_list:
                result = TaskResult(task_index=i, task_key=key)
                results.append(result)
                continue

            entity_placements = []

            # Grid area is left 355 pixels only — entities live on the right side
            grid_right = 355
            print(f"    Search area: x=[0, {grid_right}] full_image={w}x{h}")

            for edata in entity_data:
                piece_img = edata["image"]
                pw, ph = edata["size"]
                piece_resized = piece_img.convert("RGBA").resize((pw, ph), Image.LANCZOS)

                # === PASS 1: Coarse scan — only within grid area ===
                stride = max(pw // 3, 8)
                composites = []
                positions = []

                for ty in range(0, h - ph + 1, stride):
                    for tx in range(0, grid_right - pw + 1, stride):
                        # Paste piece onto FULL image (not cropped)
                        comp = img.copy().convert("RGBA")
                        comp.paste(piece_resized, (tx, ty), piece_resized)
                        # Feed full image to CLIP so it sees full context
                        composites.append(comp.convert("RGB"))
                        positions.append((tx + pw // 2, ty + ph // 2))

                print(f"    Pass 1: {len(composites)} positions (stride={stride})...")

                batch_size = 32
                scores = []
                for s in range(0, len(composites), batch_size):
                    e = min(s + batch_size, len(composites))
                    feats = self.solver.encode_images_batch(composites[s:e])
                    g = (feats @ good_mean.T).squeeze(-1)
                    b = (feats @ bad_mean.T).squeeze(-1)
                    sc = (g - b).cpu().tolist()
                    if isinstance(sc, float):
                        scores.append(sc)
                    else:
                        scores.extend(sc)

                scores = np.array(scores)
                top_k = min(20, len(scores))
                top_indices = np.argsort(scores)[-top_k:]
                best_coarse = positions[top_indices[-1]]
                print(f"    Pass 1 best: {best_coarse} score={scores[top_indices[-1]]:.4f}")

                # === PASS 2: Fine scan around top 5 positions ===
                fine_composites = []
                fine_positions = []

                for tidx in top_indices[-5:]:
                    cx, cy = positions[tidx]
                    tlx, tly = cx - pw // 2, cy - ph // 2

                    for fy in range(max(0, tly - stride), min(h - ph + 1, tly + stride + 1), 2):
                        for fx in range(max(0, tlx - stride), min(grid_right - pw + 1, tlx + stride + 1), 2):
                            comp = img.copy().convert("RGBA")
                            comp.paste(piece_resized, (fx, fy), piece_resized)
                            fine_composites.append(comp.convert("RGB"))
                            fine_positions.append((fx + pw // 2, fy + ph // 2))

                print(f"    Pass 2: {len(fine_composites)} positions (stride=2)...")

                fine_scores = []
                for s in range(0, len(fine_composites), batch_size):
                    e = min(s + batch_size, len(fine_composites))
                    feats = self.solver.encode_images_batch(fine_composites[s:e])
                    g = (feats @ good_mean.T).squeeze(-1)
                    b = (feats @ bad_mean.T).squeeze(-1)
                    sc = (g - b).cpu().tolist()
                    if isinstance(sc, float):
                        fine_scores.append(sc)
                    else:
                        fine_scores.extend(sc)

                fine_scores = np.array(fine_scores)
                best_idx = int(np.argmax(fine_scores))
                drop_x, drop_y = fine_positions[best_idx]
                best_score = float(fine_scores[best_idx])

                print(f"    → Entity {edata['index']}: drop=({drop_x},{drop_y}) score={best_score:.4f}")

                entity_placements.append({
                    "entity_id": edata["id"],
                    "start_x": edata["start_coords"][0],
                    "start_y": edata["start_coords"][1],
                    "drop_x": drop_x,
                    "drop_y": drop_y,
                    "score": best_score,
                    "size": edata["size"],
                })

            self._last_entity_placements[i] = entity_placements

            result = TaskResult(
                task_index=i,
                task_key=key,
                selected=True,
                confidence=np.mean([p["score"] for p in entity_placements]) if entity_placements else 0,
                similarity_score=np.mean([p["score"] for p in entity_placements]) if entity_placements else 0,
                label=target or "puzzle piece",
            )
            if entity_placements:
                p0 = entity_placements[0]
                result.drag_start = (p0["start_x"], p0["start_y"])
                result.drag_end = (p0["drop_x"], p0["drop_y"])

            results.append(result)
            self._annotate_entity_drag(img, entity_images_list, entity_placements, question, i)

        answers = {}
        for i, result in enumerate(results):
            task = tasklist[i] if i < len(tasklist) else {}
            task_key = result.task_key

            if task.get("entities") and i in self._last_entity_placements:
                entity_answers = {}
                for placement in self._last_entity_placements[i]:
                    dx = placement["drop_x"]
                    dy = placement["drop_y"]
                    if is_normalized and i < len(images):
                        dx = round(dx / images[i].size[0], 4)
                        dy = round(dy / images[i].size[1], 4)
                    entity_answers[placement["entity_id"]] = {"x": dx, "y": dy}
                answers[task_key] = json.dumps(entity_answers, separators=(",", ":"))

        solution = SolutionResult(
            challenge_type=ChallengeType.DRAG_DROP,
            question=question,
            tasks=results,
            overall_confidence=np.mean([r.confidence for r in results]) if results else 0,
            solved=False,
            annotated_image_path=os.path.join(self.output_dir, "solution_drag_entity_0.png"),
            answer_payload=answers,
        )
        self._save_solution_data(solution, challenge)
        return solution

    def _solve_grid_challenge(
        self,
        challenge: Dict,
        images: List[Image.Image],
        task_keys: List[str],
        target: str,
        question: str,
    ) -> SolutionResult:
        """Solve grid-based classification challenge."""
        print(f"\n[6] CLIP Grid Solver")

        results = []

        # Detect grid dimensions from image aspect ratio or task metadata
        for i, (img, key) in enumerate(zip(images, task_keys)):
            print(f"\n  --- Task {i} ---")

            w, h = img.size
            aspect = w / h if h > 0 else 1

            # Guess grid size
            if aspect > 1.3:
                grid_cols, grid_rows = 4, 3
            elif aspect < 0.7:
                grid_cols, grid_rows = 3, 4
            else:
                grid_cols, grid_rows = 3, 3

            result = self.solver.solve_grid_classification(
                img, key, target, grid_rows, grid_cols
            )
            result.task_index = i
            results.append(result)

            self.annotator.annotate_grid(
                img, result, question, grid_rows, grid_cols
            )

        # Build answer payload
        answers = {}
        for result in results:
            answers[result.task_key] = {
                "selected_cells": result.grid_cells,
                "selected": result.selected,
            }

        solution = SolutionResult(
            challenge_type=ChallengeType.GRID_CLASSIFICATION,
            question=question,
            tasks=results,
            overall_confidence=np.mean([r.confidence for r in results]),
            solved=True,
            annotated_image_path=os.path.join(self.output_dir, "solution_grid.png"),
            answer_payload=answers,
        )

        self._save_solution_data(solution, challenge)
        return solution

    # ── Solution data persistence ─────────────────────────────────────

    def _save_solution_data(self, solution: SolutionResult, challenge: Dict):
        """Save comprehensive solution data to JSON."""
        ts = int(time.time())

        # Save raw challenge
        raw_path = os.path.join(self.output_dir, f"challenge_raw_{ts}.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(challenge, f, indent=2, default=str)

        # Save solution summary
        solution_data = {
            "timestamp": ts,
            "challenge_type": solution.challenge_type.value,
            "question": solution.question,
            "overall_confidence": solution.overall_confidence,
            "solved": solution.solved,
            "annotated_image": solution.annotated_image_path,
            "answer_payload": solution.answer_payload,
            "tasks": [],
        }

        for task in solution.tasks:
            task_data = {
                "index": task.task_index,
                "key": task.task_key,
                "selected": task.selected,
                "confidence": task.confidence,
                "similarity_score": task.similarity_score,
                "label": task.label,
            }
            if task.click_point:
                task_data["click_point"] = {"x": task.click_point[0], "y": task.click_point[1]}
            if task.drag_start:
                task_data["drag_start"] = {"x": task.drag_start[0], "y": task.drag_start[1]}
            if task.drag_end:
                task_data["drag_end"] = {"x": task.drag_end[0], "y": task.drag_end[1]}
            if task.grid_cells:
                task_data["grid_cells"] = task.grid_cells

            solution_data["tasks"].append(task_data)

        sol_path = os.path.join(self.output_dir, f"solution_{ts}.json")
        with open(sol_path, "w", encoding="utf-8") as f:
            json.dump(solution_data, f, indent=2)

        print(f"\n  [Save] Raw challenge: {raw_path}")
        print(f"  [Save] Solution: {sol_path}")

    # ── Save individual task images ───────────────────────────────────

    def _save_task_images(
        self,
        images: List[Image.Image],
        task_keys: List[str],
        results: List[TaskResult],
    ):
        """Save individual task images with annotations."""
        ts = int(time.time())
        img_dir = os.path.join(self.output_dir, f"tasks_{ts}")
        os.makedirs(img_dir, exist_ok=True)

        for i, (img, key, result) in enumerate(zip(images, task_keys, results)):
            annotated = img.copy().convert("RGBA")
            draw = ImageDraw.Draw(annotated)

            if result.selected:
                # Green border for selected
                overlay = Image.new("RGBA", annotated.size, (0, 200, 0, 50))
                annotated = Image.alpha_composite(annotated, overlay)
                draw = ImageDraw.Draw(annotated)
                w, h = annotated.size
                for b in range(5):
                    draw.rectangle([(b, b), (w-1-b, h-1-b)], outline=(0, 255, 0))
            else:
                # Red border for rejected
                overlay = Image.new("RGBA", annotated.size, (200, 0, 0, 30))
                annotated = Image.alpha_composite(annotated, overlay)
                draw = ImageDraw.Draw(annotated)
                w, h = annotated.size
                for b in range(3):
                    draw.rectangle([(b, b), (w-1-b, h-1-b)], outline=(200, 60, 60))

            if result.click_point:
                cx, cy = result.click_point
                draw.ellipse([(cx-15, cy-15), (cx+15, cy+15)], outline=(255, 40, 40), width=3)
                draw.ellipse([(cx-3, cy-3), (cx+3, cy+3)], fill=(255, 40, 40))

            if result.drag_end:
                ex, ey = result.drag_end
                draw.rectangle(
                    [(ex-20, ey-20), (ex+20, ey+20)],
                    outline=(0, 255, 0),
                    width=3,
                )
                if result.drag_start:
                    sx, sy = result.drag_start
                    draw.line([(sx, sy), (ex, ey)], fill=(255, 220, 0), width=2)

            fname = f"task_{i:02d}_{'MATCH' if result.selected else 'skip'}_{result.similarity_score:.3f}.png"
            annotated.save(os.path.join(img_dir, fname), "PNG")

        print(f"  [Save] Individual tasks: {img_dir}/")

    # ── Answer submission ─────────────────────────────────────────────

    def _submit_answers(
        self,
        challenge: Dict,
        solution: SolutionResult,
        c_data: Dict,
        proof: str,
    ) -> Optional[Dict]:
        """
        Submit the solved answers back to hCaptcha.
        """
        print(f"\n[7] Submitting answers...")

        challenge_key = challenge.get("key", "")
        tasklist = challenge.get("tasklist", [])

        if not challenge_key:
            print("    ✗ No challenge key!")
            return None

        answers = solution.answer_payload

        # Generate fresh motion data for submission
        motion = self._gen_answer_motion_data(solution)

        form = {
            "v": self.version or "",
            "job_mode": challenge.get("request_type", "image_label_binary"),
            "answers": json.dumps(answers, separators=(",", ":")),
            "serverdomain": self.host,
            "sitekey": self.site_key,
            "motionData": json.dumps(motion, separators=(",", ":")),
            "n": proof,
            "c": json.dumps(c_data, separators=(",", ":")),
        }

        try:
            resp = self.session.post(
                f"{self.HC_API}/checkcaptcha/{self.site_key}/{challenge_key}",
                data=form,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "Origin": self.HC_ASSETS,
                    "Referer": f"{self.HC_ASSETS}/captcha/v1/{self.version}/static/hcaptcha.html",
                    "Sec-Fetch-Dest": "empty",
                    "Sec-Fetch-Mode": "cors",
                    "Sec-Fetch-Site": "same-site",
                },
                timeout=30,
            )

            result = resp.json()

            if result.get("pass"):
                token = result.get("generated_pass_UUID", "")
                print(f"    ✓ CAPTCHA SOLVED!")
                print(f"    Token: {token[:50]}...")
                return result
            else:
                print(f"    ✗ Rejected: {result.get('error-codes', result)}")
                return result

        except Exception as e:
            print(f"    ✗ Submission error: {e}")
            return None

    def _gen_answer_motion_data(self, solution: SolutionResult) -> Dict:
        """
        Generate realistic mouse motion data for the answer submission.
        Simulates the user clicking through the challenge images.
        """
        now = int(time.time() * 1000)
        challenge_start = now - random.randint(8000, 25000)
        wid = self._gen_widget_id()

        mm = []
        md = []
        mu = []

        t = challenge_start + random.randint(500, 2000)
        x, y = random.randint(100, 300), random.randint(100, 300)

        for task in solution.tasks:
            if task.selected or task.click_point or task.drag_end:
                # Move to task image area
                target_x = random.randint(50, 400)
                target_y = random.randint(100, 350)

                # Generate bezier-like path
                steps = random.randint(8, 25)
                for s in range(steps):
                    progress = (s + 1) / steps
                    # Ease-in-out curve
                    ease = progress * progress * (3 - 2 * progress)
                    cx = int(x + (target_x - x) * ease + random.randint(-3, 3))
                    cy = int(y + (target_y - y) * ease + random.randint(-3, 3))
                    t += random.randint(8, 40)
                    mm.append([cx, cy, t])

                x, y = target_x, target_y

                # Click
                t += random.randint(30, 150)
                md.append([x + random.randint(-2, 2), y + random.randint(-2, 2), t])
                t += random.randint(60, 180)
                mu.append([x + random.randint(-2, 2), y + random.randint(-2, 2), t])

                # Pause between tasks
                t += random.randint(300, 1200)

        # Move to verify button
        verify_x = random.randint(180, 250)
        verify_y = random.randint(420, 460)
        steps = random.randint(10, 20)
        for s in range(steps):
            progress = (s + 1) / steps
            ease = progress * progress * (3 - 2 * progress)
            cx = int(x + (verify_x - x) * ease + random.randint(-2, 2))
            cy = int(y + (verify_y - y) * ease + random.randint(-2, 2))
            t += random.randint(10, 35)
            mm.append([cx, cy, t])

        x, y = verify_x, verify_y
        t += random.randint(50, 200)
        md.append([x, y, t])
        t += random.randint(80, 200)
        mu.append([x, y, t])

        return {
            "st": challenge_start,
            "dct": challenge_start,
            "mm": mm,
            "mm-mp": 0,
            "md": md,
            "md-mp": 0,
            "mu": mu,
            "mu-mp": 0,
            "v": 1,
            "topLevel": {
                "st": challenge_start - random.randint(3000, 8000),
                "sc": {
                    "availWidth": 1920, "availHeight": 1040,
                    "width": 1920, "height": 1080,
                    "colorDepth": 24, "pixelDepth": 24,
                    "availLeft": 0, "availTop": 0,
                },
                "nv": {
                    "vendorSub": "",
                    "productSub": "20030107",
                    "vendor": "Google Inc.",
                    "maxTouchPoints": 0,
                    "hardwareConcurrency": 8,
                    "cookieEnabled": True,
                    "appCodeName": "Mozilla",
                    "appName": "Netscape",
                    "appVersion": "5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
                    "platform": "Win32",
                    "product": "Gecko",
                    "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
                    "language": "en-US",
                    "languages": ["en-US", "en"],
                    "onLine": True,
                    "webdriver": False,
                    "pdfViewerEnabled": True,
                    "deviceMemory": 8,
                    "plugins": [],
                },
                "dr": "",
                "inv": False,
                "exec": False,
                "wn": [[1366, 768, challenge_start - 2000]],
                "wn-mp": 0,
                "xy": [[0, 0, challenge_start - 2000]],
                "xy-mp": 0,
                "mm": [],
                "mm-mp": 0,
            },
            "session": [],
            "widgetList": [wid],
            "widgetId": wid,
            "href": self.site_url,
            "prev": {
                "escaped": False,
                "passed": False,
                "expiredChallenge": False,
                "expiredResponse": False,
            },
        }

    # ── Composite overview image ──────────────────────────────────────

    def _create_overview_image(
        self,
        challenge: Dict,
        solution: SolutionResult,
        images: List[Image.Image],
    ) -> str:
        """
        Create a comprehensive overview image showing:
        - Challenge question
        - All images with annotations
        - Solution summary
        - Confidence metrics
        """
        n = len(images)
        if n == 0:
            return ""

        cell_size = 200
        padding = 12
        cols = min(n, 5)
        rows = math.ceil(n / cols)
        header_h = 100
        stats_h = 120
        footer_h = 50

        total_w = max(cols * (cell_size + padding) + padding, 600)
        total_h = header_h + rows * (cell_size + padding) + padding + stats_h + footer_h

        canvas = Image.new("RGBA", (total_w, total_h), (25, 25, 35, 255))
        draw = ImageDraw.Draw(canvas)

        # ── Header ──
        draw.rectangle([(0, 0), (total_w, header_h)], fill=(15, 15, 25, 255))

        # Title
        draw.text(
            (15, 10),
            "hCaptcha CLIP AI Solution",
            fill=(100, 200, 255),
            font=self.annotator.font_large,
        )

        # Challenge type badge
        type_text = f"[{solution.challenge_type.value.upper()}]"
        draw.text(
            (15, 40),
            type_text,
            fill=(255, 180, 0),
            font=self.annotator.font_medium,
        )

        # Question
        q_wrapped = self.annotator._wrap_text(solution.question, total_w - 30, self.annotator.font_medium)
        y_q = 65
        for line in q_wrapped[:2]:
            draw.text((15, y_q), line, fill=(220, 220, 220), font=self.annotator.font_medium)
            y_q += 18

        # ── Image grid ──
        for i in range(min(n, len(solution.tasks))):
            task = solution.tasks[i]
            img = images[i] if i < len(images) else None
            if not img:
                continue

            row = i // cols
            col = i % cols
            x = padding + col * (cell_size + padding)
            y = header_h + padding + row * (cell_size + padding)

            # Resize
            cell = img.copy().convert("RGBA")
            cell = cell.resize((cell_size, cell_size), Image.LANCZOS)

            # Apply overlay based on result
            if task.selected:
                overlay = Image.new("RGBA", (cell_size, cell_size), (0, 200, 0, 60))
                cell = Image.alpha_composite(cell, overlay)
                cell_draw = ImageDraw.Draw(cell)
                for b in range(4):
                    cell_draw.rectangle(
                        [(b, b), (cell_size - 1 - b, cell_size - 1 - b)],
                        outline=(0, 255, 0),
                    )
                # Checkmark icon
                self.annotator._draw_checkmark(cell_draw, cell_size - 30, 8, 18, (0, 255, 0))
            else:
                overlay = Image.new("RGBA", (cell_size, cell_size), (180, 0, 0, 40))
                cell = Image.alpha_composite(cell, overlay)
                cell_draw = ImageDraw.Draw(cell)
                for b in range(2):
                    cell_draw.rectangle(
                        [(b, b), (cell_size - 1 - b, cell_size - 1 - b)],
                        outline=(120, 120, 120),
                    )
                self.annotator._draw_xmark(cell_draw, cell_size - 25, 10, 14, (180, 60, 60))

            # Click point annotation
            if task.click_point:
                cx, cy = task.click_point
                # Scale coordinates to cell size
                if i < len(images):
                    ow, oh = images[i].size
                    scaled_cx = int(cx * cell_size / ow)
                    scaled_cy = int(cy * cell_size / oh)
                else:
                    scaled_cx, scaled_cy = cx, cy

                cell_draw = ImageDraw.Draw(cell)
                r = 12
                cell_draw.ellipse(
                    [(scaled_cx - r, scaled_cy - r), (scaled_cx + r, scaled_cy + r)],
                    outline=(255, 40, 40),
                    width=3,
                )
                cell_draw.ellipse(
                    [(scaled_cx - 3, scaled_cy - 3), (scaled_cx + 3, scaled_cy + 3)],
                    fill=(255, 40, 40),
                )

            # Drag target annotation
            if task.drag_end:
                ex, ey = task.drag_end
                if i < len(images):
                    ow, oh = images[i].size
                    scaled_ex = int(ex * cell_size / ow)
                    scaled_ey = int(ey * cell_size / oh)
                else:
                    scaled_ex, scaled_ey = ex, ey

                cell_draw = ImageDraw.Draw(cell)
                box_r = 15
                cell_draw.rectangle(
                    [(scaled_ex - box_r, scaled_ey - box_r),
                     (scaled_ex + box_r, scaled_ey + box_r)],
                    outline=(0, 255, 0),
                    width=3,
                )

                if task.drag_start:
                    sx, sy = task.drag_start
                    if i < len(images):
                        scaled_sx = int(sx * cell_size / ow)
                        scaled_sy = int(sy * cell_size / oh)
                    else:
                        scaled_sx, scaled_sy = sx, sy
                    self.annotator._draw_arrow(
                        cell_draw, scaled_sx, scaled_sy, scaled_ex, scaled_ey,
                        color=(255, 220, 0), width=2,
                    )

            # Score badge
            score_text = f"{task.similarity_score:.3f}"
            bg = (0, 160, 0) if task.selected else (160, 40, 40)
            self.annotator._draw_label(
                ImageDraw.Draw(cell), 4, cell_size - 22,
                score_text, bg_color=bg, font=self.annotator.font_tiny,
            )

            # Index badge
            self.annotator._draw_label(
                ImageDraw.Draw(cell), 4, 4,
                f"#{i}", bg_color=(50, 50, 50), font=self.annotator.font_tiny,
            )

            canvas.paste(cell, (x, y))

        # ── Statistics panel ──
        stats_y = header_h + rows * (cell_size + padding) + padding + 5
        draw.rectangle([(0, stats_y), (total_w, stats_y + stats_h)], fill=(20, 20, 30, 255))

        selected_count = sum(1 for t in solution.tasks if t.selected)
        total_count = len(solution.tasks)
        scores = [t.similarity_score for t in solution.tasks]
        confidences = [t.confidence for t in solution.tasks]

        stats_lines = [
            f"Selected: {selected_count}/{total_count}",
            f"Confidence: {solution.overall_confidence:.4f}",
            f"Score Range: [{min(scores):.3f} — {max(scores):.3f}]" if scores else "No scores",
            f"Mean Score: {np.mean(scores):.4f}" if scores else "",
            f"Score Std: {np.std(scores):.4f}" if scores else "",
        ]

        col1_x = 15
        col2_x = total_w // 2 + 10
        y_stat = stats_y + 10

        for j, line in enumerate(stats_lines):
            x_pos = col1_x if j < 3 else col2_x
            y_pos = y_stat + (j % 3) * 20 if j < 3 else y_stat + ((j - 3) % 3) * 20
            color = (200, 200, 200) if "Selected" not in line else (100, 255, 100)
            draw.text((x_pos, y_pos), line, fill=color, font=self.annotator.font_small)

        # Score bar visualization
        bar_y = stats_y + 75
        bar_h = 20
        bar_w = total_w - 30
        draw.rectangle([(15, bar_y), (15 + bar_w, bar_y + bar_h)], fill=(40, 40, 50))

        if scores:
            for j, score in enumerate(scores):
                bar_x = 15 + int((j / max(len(scores) - 1, 1)) * (bar_w - 10))
                dot_color = (0, 255, 0) if solution.tasks[j].selected else (255, 60, 60)
                dot_y = bar_y + bar_h - int(score * bar_h)
                draw.ellipse(
                    [(bar_x - 3, dot_y - 3), (bar_x + 3, dot_y + 3)],
                    fill=dot_color,
                )

        draw.text((15, bar_y + bar_h + 2), "Score Distribution", fill=(140, 140, 140), font=self.annotator.font_tiny)

        # ── Footer ──
        footer_y = total_h - footer_h
        draw.rectangle([(0, footer_y), (total_w, total_h)], fill=(10, 10, 15, 255))
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        draw.text(
            (15, footer_y + 15),
            f"Generated: {timestamp_str} | Model: {self.solver.model_name} | Device: {self.solver.device}",
            fill=(100, 100, 120),
            font=self.annotator.font_tiny,
        )

        out_path = os.path.join(self.output_dir, "solution_overview.png")
        canvas.save(out_path, "PNG")
        print(f"\n  [Overview] Saved: {out_path}")
        return out_path

    # ── Main entry point ──────────────────────────────────────────────

    def solve(self, submit: bool = False) -> Optional[SolutionResult]:
        """
        Full pipeline: fetch challenge → solve with CLIP → annotate → optionally submit.

        Args:
            submit: If True, submit the answers to hCaptcha for verification.

        Returns:
            SolutionResult with all data, or None on failure.
        """
        print(f"\n{'═'*60}")
        print(f"  hCaptcha CLIP AI Solver — Full Pipeline")
        print(f"{'═'*60}")
        print(f"  Site: {self.site_url}")
        print(f"  Key: {self.site_key}")
        print(f"  Model: {self.solver.model_name}")
        print(f"  Output: {self.output_dir}")
        print(f"{'═'*60}")
        self._rotate_proxy()

        # Step 1: Site config
        config = self._check_site_config()
        if not config:
            return None

        # Step 2: Check HSW
        print("\n[2] Checking hsw.js...")
        if not os.path.exists("hsw.js"):
            print("    ✗ hsw.js not found! Place hsw.js in the working directory.")
            return None
        self.hsw_path = "hsw.js"
        print("    ✓ Found hsw.js")

        c_data = config.get("c", {})
        max_rounds = 10

        for rnd in range(1, max_rounds + 1):
            jwt_req = c_data.get("req", "")
            if not jwt_req:
                print(f"    ✗ No JWT in round {rnd}")
                return None

            print(f"\n[3] Proof round {rnd}/{max_rounds}...")
            proof = self._solve_hsw(jwt_req)
            if not proof:
                return None

            print(f"\n[4] getcaptcha round {rnd}...")
            try:
                challenge = self._get_challenge(c_data, proof)
            except Exception as e:
                print(f"    ✗ {e}")
                return None

            # Got a challenge with images
            if challenge.get("key") and challenge.get("tasklist"):
                q = challenge.get("requester_question", {})
                if isinstance(q, dict):
                    q = q.get("en", "")
                n_tasks = len(challenge["tasklist"])
                print(f"    ✓ Challenge received!")
                print(f"    Question: {q}")
                print(f"    Tasks: {n_tasks}")
                print(f"    Request type: {challenge.get('request_type', 'N/A')}")

                # Save raw challenge
                raw_path = os.path.join(self.output_dir, "challenge_raw.json")
                with open(raw_path, "w", encoding="utf-8") as f:
                    json.dump(challenge, f, indent=2, default=str)

                # ── SOLVE WITH CLIP ──
                t0 = time.time()
                solution = self._solve_challenge(challenge)
                solve_time = time.time() - t0

                if solution:
                    print(f"\n{'─'*60}")
                    print(f"  SOLUTION COMPLETE ({solve_time:.1f}s)")
                    print(f"  Type: {solution.challenge_type.value}")
                    print(f"  Confidence: {solution.overall_confidence:.4f}")

                    selected = sum(1 for t in solution.tasks if t.selected)
                    print(f"  Selected: {selected}/{len(solution.tasks)}")

                    # Load images again for overview
                    images, task_keys, _ = self._load_images_from_challenge(challenge)
                    if images:
                        # Save individual annotated task images
                        self._save_task_images(images, task_keys, solution.tasks)

                        # Create overview composite
                        overview_path = self._create_overview_image(
                            challenge, solution, images
                        )
                        solution.annotated_image_path = overview_path

                    # Optionally submit
                    if submit:
                        submit_c = challenge.get("c", c_data)
                        submit_jwt = submit_c.get("req", jwt_req)

                        print(f"\n[8] Generating submission proof...")
                        submit_proof = self._solve_hsw(submit_jwt)
                        if submit_proof:
                            result = self._submit_answers(
                                challenge, solution, submit_c, submit_proof
                            )
                            print(f"\n  [checkcaptcha response]:")
                            print(f"  {json.dumps(result, indent=2, default=str)}")
                            if result and result.get("pass"):
                                solution.solved = True
                                print(f"\n  ✓ CAPTCHA FULLY SOLVED AND VERIFIED!")
                                print(f"  Token: {result.get('generated_pass_UUID', 'N/A')}")
                            else:
                                solution.solved = False
                                print(f"\n  ✗ Answer REJECTED by hCaptcha")
                        else:
                            print(f"    ✗ Failed to generate submission proof")
                            solution.solved = False

                    print(f"\n{'═'*60}")
                    print(f"  All results saved to: {self.output_dir}/")
                    print(f"{'═'*60}\n")

                    return solution
                else:
                    print(f"    ✗ CLIP solver returned no solution")
                    return None

            # Auto-pass
            if challenge.get("generated_pass_UUID"):
                print("    ✓ Auto-passed (no challenge needed)")
                return SolutionResult(
                    challenge_type=ChallengeType.UNKNOWN,
                    question="auto-pass",
                    tasks=[],
                    overall_confidence=1.0,
                    solved=True,
                )

            # New JWT = retry
            if (
                challenge.get("c")
                and challenge["c"].get("req")
                and challenge["c"]["req"] != jwt_req
            ):
                print(f"    ↻ New proof requested, retrying...")
                c_data = challenge["c"]
                continue

            # Error
            print(f"    ✗ Failed: {challenge.get('error-codes', '?')}")
            self._dbg(json.dumps(challenge, indent=2)[:500])
            return None

        print(f"    ✗ Max rounds ({max_rounds}) reached without getting challenge")
        return None

    def solve_from_saved(self, challenge_json_path: str) -> Optional[SolutionResult]:
        """
        Solve a previously saved challenge JSON (offline mode).
        Useful for testing/tuning without hitting hCaptcha again.
        """
        print(f"\n{'═'*60}")
        print(f"  Offline CLIP Solver — From Saved Challenge")
        print(f"{'═'*60}")

        with open(challenge_json_path, "r", encoding="utf-8") as f:
            challenge = json.load(f)

        solution = self._solve_challenge(challenge)

        if solution:
            images, task_keys, _ = self._load_images_from_challenge(challenge)
            if images:
                self._save_task_images(images, task_keys, solution.tasks)
                self._create_overview_image(challenge, solution, images)

        return solution

    def batch_solve(
        self,
        num_attempts: int = 5,
        submit: bool = False,
        delay_range: Tuple[float, float] = (3.0, 8.0),
    ) -> List[SolutionResult]:
        """
        Attempt to solve multiple challenges in sequence.
        Useful for testing accuracy across different challenge types.
        """
        print(f"\n{'═'*60}")
        print(f"  Batch Solver — {num_attempts} attempts")
        print(f"{'═'*60}")

        results = []
        success = 0
        fail = 0

        for attempt in range(1, num_attempts + 1):
            print(f"\n\n{'▓'*60}")
            print(f"  ATTEMPT {attempt}/{num_attempts}")
            print(f"{'▓'*60}")

            # Create per-attempt output directory
            attempt_dir = os.path.join(self.output_dir, f"attempt_{attempt:03d}")
            original_dir = self.output_dir
            self.output_dir = attempt_dir
            self.annotator = SolutionAnnotator(attempt_dir)
            os.makedirs(attempt_dir, exist_ok=True)

            try:
                solution = self.solve(submit=submit)

                if solution:
                    results.append(solution)
                    if solution.solved:
                        success += 1
                        print(f"\n  ✓ Attempt {attempt}: SOLVED (conf={solution.overall_confidence:.3f})")
                    else:
                        fail += 1
                        print(f"\n  ~ Attempt {attempt}: Solved but not verified")
                else:
                    fail += 1
                    print(f"\n  ✗ Attempt {attempt}: FAILED")

            except Exception as e:
                fail += 1
                print(f"\n  ✗ Attempt {attempt}: ERROR — {e}")
                import traceback
                traceback.print_exc()

            finally:
                self.output_dir = original_dir
                self.annotator = SolutionAnnotator(original_dir)

            # Delay between attempts
            if attempt < num_attempts:
                delay = random.uniform(*delay_range)
                print(f"\n  Waiting {delay:.1f}s before next attempt...")
                time.sleep(delay)

        # Summary
        print(f"\n\n{'═'*60}")
        print(f"  BATCH SUMMARY")
        print(f"{'═'*60}")
        print(f"  Total: {num_attempts}")
        print(f"  Success: {success}")
        print(f"  Failed: {fail}")
        print(f"  Rate: {success/max(num_attempts,1)*100:.1f}%")

        if results:
            confs = [r.overall_confidence for r in results]
            print(f"  Avg Confidence: {np.mean(confs):.4f}")
            print(f"  Min Confidence: {np.min(confs):.4f}")
            print(f"  Max Confidence: {np.max(confs):.4f}")

            # Type breakdown
            type_counts = {}
            for r in results:
                t = r.challenge_type.value
                type_counts[t] = type_counts.get(t, 0) + 1
            print(f"  Challenge types: {type_counts}")

        print(f"{'═'*60}\n")

        # Save batch summary
        summary = {
            "total_attempts": num_attempts,
            "success": success,
            "failed": fail,
            "success_rate": success / max(num_attempts, 1),
            "results": [
                {
                    "type": r.challenge_type.value,
                    "question": r.question,
                    "confidence": r.overall_confidence,
                    "solved": r.solved,
                    "num_tasks": len(r.tasks),
                    "selected": sum(1 for t in r.tasks if t.selected),
                }
                for r in results
            ],
        }
        with open(os.path.join(self.output_dir, "batch_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return results


# ══════════════════════════════════════════════════════════════════════
# CLI & MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="hCaptcha CLIP AI Solver — Solve any hCaptcha challenge using local CLIP AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve a single challenge (download + annotate only)
  python solver.py --site-key YOUR_KEY --site-url https://example.com

  # Solve and submit the answer
  python solver.py --site-key YOUR_KEY --site-url https://example.com --submit

  # Batch solve 10 challenges
  python solver.py --site-key YOUR_KEY --site-url https://example.com --batch 10

  # Use a specific CLIP model
  python solver.py --site-key YOUR_KEY --site-url https://example.com --model ViT-B/32

  # Solve from a saved challenge JSON
  python solver.py --offline challenge_raw.json

  # Force CPU
  python solver.py --site-key YOUR_KEY --site-url https://example.com --device cpu
        """,
    )

    parser.add_argument("--site-key", type=str, default="58366d97-3e8c-4b57-a679-4a41c8423be3",
                        help="hCaptcha site key")
    parser.add_argument("--site-url", type=str, default="https://nopecha.com/captcha/hcaptcha",
                        help="Target site URL")
    parser.add_argument("--output", type=str, default="hcaptcha_solutions",
                        help="Output directory for solutions")
    parser.add_argument("--model", type=str, default="ViT-L/14@336px",
                        choices=["ViT-L/14@336px", "ViT-L/14", "ViT-B/16", "ViT-B/32",
                                 "RN50x64", "RN50x16", "RN50x4", "RN101", "RN50"],
                        help="CLIP model to use (default: ViT-L/14@336px — best accuracy)")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu", "mps"],
                        help="Compute device (auto-detected if not specified)")
    parser.add_argument("--submit", action="store_true",
                        help="Submit answers to hCaptcha for verification")
    parser.add_argument("--batch", type=int, default=0,
                        help="Number of challenges to solve in batch mode")
    parser.add_argument("--offline", type=str, default=None,
                        help="Path to saved challenge JSON for offline solving")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")

    args = parser.parse_args()

    # Create solver
    solver = HCaptchaCLIPSolver(
        site_key=args.site_key,
        site_url=args.site_url,
        output_dir=args.output,
        debug=args.debug,
        clip_model=args.model,
        device=args.device,
    )

    if args.offline:
        # Offline mode: solve from saved JSON
        result = solver.solve(submit=True)
        if result:
            print(f"\n✓ Offline solution complete!")
            print(f"  Type: {result.challenge_type.value}")
            print(f"  Confidence: {result.overall_confidence:.4f}")
        else:
            print(f"\n✗ Offline solving failed")

    elif args.batch > 0:
        # Batch mode
        results = solver.batch_solve(
            num_attempts=args.batch,
            submit=args.submit,
        )

    else:
        # Single solve
        result = solver.solve(submit=args.submit)
        if result:
            print(f"\n✓ Solution complete!")
            print(f"  Type: {result.challenge_type.value}")
            print(f"  Question: {result.question}")
            print(f"  Confidence: {result.overall_confidence:.4f}")
            print(f"  Solved: {result.solved}")
            selected = sum(1 for t in result.tasks if t.selected)
            print(f"  Selected: {selected}/{len(result.tasks)}")
        else:
            print(f"\n✗ Solving failed")


if __name__ == "__main__":
    main()
