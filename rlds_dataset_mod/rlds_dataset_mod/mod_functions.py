from abc import ABC, abstractmethod

import uuid
import dlimp as dl
import tensorflow as tf
import tensorflow_datasets as tfds


class TfdsModFunction(ABC):
    @classmethod
    @abstractmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        """
        Modifies the data builder feature dict to reflect feature changes of ModFunction.
        """
        ...

    @classmethod
    @abstractmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        """
        Perform arbitrary modifications on the dataset that comply with the modified feature definition.
        """
        ...


def mod_obs_features(features, obs_feature_mod_function):
    """Utility function to only modify keys in observation dict."""
    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": tfds.features.FeaturesDict(
                        {
                            key: obs_feature_mod_function(
                                key, features["steps"]["observation"][key]
                            )
                            for key in features["steps"]["observation"].keys()
                        }
                    ),
                    **{
                        key: features["steps"][key]
                        for key in features["steps"].keys()
                        if key not in ("observation",)
                    },
                }
            ),
            **{key: features[key] for key in features.keys() if key not in ("steps",)},
        }
    )


class ResizeAndJpegEncode(TfdsModFunction):
    MAX_RES: int = 256

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        def downsize_and_jpeg(key, feat):
            """Downsizes image features, encodes as jpeg."""
            if len(feat.shape) >= 2 and feat.shape[0] >= 64 and feat.shape[1] >= 64:  # is image / depth feature
                should_jpeg_encode = (
                    isinstance(feat, tfds.features.Image) and "depth" not in key
                )
                if len(feat.shape) > 2:
                    new_shape = (ResizeAndJpegEncode.MAX_RES, ResizeAndJpegEncode.MAX_RES, feat.shape[2])
                else:
                    new_shape = (ResizeAndJpegEncode.MAX_RES, ResizeAndJpegEncode.MAX_RES)

                if isinstance(feat, tfds.features.Image):
                    return tfds.features.Image(
                        shape=new_shape,
                        dtype=feat.dtype,
                        encoding_format="jpeg" if should_jpeg_encode else "png",
                        doc=feat.doc,
                    )
                else:
                    return tfds.features.Tensor(
                        shape=new_shape,
                        dtype=feat.dtype,
                        doc=feat.doc,
                    )

            return feat

        return mod_obs_features(features, downsize_and_jpeg)

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def resize_image_fn(step):
            # resize images
            for key in step["observation"]:
                if len(step["observation"][key].shape) >= 2 and (
                    step["observation"][key].shape[0] >= 64
                    or step["observation"][key].shape[1] >= 64
                ):
                    size = (ResizeAndJpegEncode.MAX_RES,
                            ResizeAndJpegEncode.MAX_RES)
                    if "depth" in key:
                        step["observation"][key] = tf.cast(
                            dl.utils.resize_depth_image(
                                tf.cast(step["observation"][key], tf.float32), size
                            ),
                            step["observation"][key].dtype,
                        )
                    else:
                        step["observation"][key] = tf.cast(
                            dl.utils.resize_image(step["observation"][key], size),
                            tf.uint8,
                        )
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(resize_image_fn)
            return episode

        return ds.map(episode_map_fn)


class FilterSuccess(TfdsModFunction):
    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        return features  # no feature changes

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.filter(lambda e: e["success"])


class FlipImgChannels(TfdsModFunction):
    FLIP_KEYS = ["image"]

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        return features  # no feature changes

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def flip(step):
            for key in cls.FLIP_KEYS:
                if key in step["observation"]:
                    step["observation"][key] = step["observation"][key][..., ::-1]
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(flip)
            return episode

        return ds.map(episode_map_fn)
    

class FlipWristImgChannels(FlipImgChannels):
    FLIP_KEYS = ["wrist_image", "hand_image"]


class AddEpisodeId(TfdsModFunction):
    """Modification function to add episode_id to episode_metadata."""
    
    # Class variable to store the counter dictionary
    # _file_path_to_counter = {}
    
    @classmethod
    def mod_features(cls, features: tfds.features.FeaturesDict) -> tfds.features.FeaturesDict:
        """Add episode_id to episode_metadata features."""
        # Convert the features to a dictionary representation
        features_dict = {}
        for key, feature in features.items():
            if key == 'episode_metadata' and isinstance(feature, tfds.features.FeaturesDict):
                # For episode_metadata, create a modified copy with the new field
                metadata_dict = {}
                for meta_key, meta_feature in feature.items():
                    metadata_dict[meta_key] = meta_feature
                
                # Add the new episode_id field
                # metadata_dict['episode_id'] = tfds.features.Tensor(shape=(), dtype=tf.string)
                metadata_dict['episode_id'] = tfds.features.Text(doc='Unique ID of the episode.')
                
                # Store the modified metadata dictionary
                features_dict[key] = tfds.features.FeaturesDict(metadata_dict)
            else:
                # Keep other features as they are
                features_dict[key] = feature
        
        # Create a new FeaturesDict with the modified structure
        return tfds.features.FeaturesDict(features_dict)
    
    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        """Add episode_id to each example based on file_path."""
        def add_episode_id(example):
            # Get the file_path as a string
            file_path = tf.py_function(
                lambda x: x.numpy().decode('utf-8') if isinstance(x.numpy(), bytes) else x.numpy(),
                [example['episode_metadata']['file_path']],
                tf.string
            )
            
            # Create a unique identifier by combining file_path with a UUID
            def create_unique_id(_):
                # Generate a unique ID by combining file path with a UUID
                unique_id = uuid.uuid4().hex
                return unique_id
            
            # Get the episode_id
            episode_id = tf.py_function(
                create_unique_id,
                [file_path],
                tf.string
            )
            
            # Add episode_id to the example
            example['episode_metadata']['episode_id'] = episode_id.decode() if isinstance(episode_id, bytes) else episode_id
            
            return example
        
        return ds.map(add_episode_id)

TFDS_MOD_FUNCTIONS = {
    "resize_and_jpeg_encode": ResizeAndJpegEncode,
    "filter_success": FilterSuccess,
    "flip_image_channels": FlipImgChannels,
    "flip_wrist_image_channels": FlipWristImgChannels,
    "add_episode_id": AddEpisodeId,
}


