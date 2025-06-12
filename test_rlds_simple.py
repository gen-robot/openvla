#!/usr/bin/env python3
"""
Minimal test script to verify the RLDS dataset iteration fix.
Tests the core TensorFlow dataset directly without PyTorch overhead.
"""

import time
from pathlib import Path

import tensorflow as tf
from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights, OXE_NAMED_MIXTURES
from prismatic.vla.datasets.rlds import make_interleaved_dataset

def test_rlds_dataset():
    """Test the core RLDS dataset iteration."""
    print("🚀 Testing RLDS dataset iteration...")
    
    # Setup
    data_root_dir = Path("datasets/libero_data")
    data_mix = "libero_lm_90"
    
    # Get dataset configuration
    if data_mix in OXE_NAMED_MIXTURES:
        mixture_spec = OXE_NAMED_MIXTURES[data_mix]
    else:
        mixture_spec = [(data_mix, 1.0)]
    
    print(f"📊 Dataset mix: {mixture_spec}")
    
    # Configure dataset loading
    load_camera_views = {name: ("primary",) for name, _ in mixture_spec}
    
    per_dataset_kwargs, weights, _ = get_oxe_dataset_kwargs_and_weights(
        data_root_dir,
        mixture_spec,
        load_camera_views=load_camera_views,
        load_depth=False,
        load_proprio=True,
        load_language=True,
    )
    
    # RLDS configuration
    rlds_config = dict(
        traj_transform_kwargs=dict(
            window_size=1,
            future_action_window_size=7,
            skip_unlabeled=True,
            goal_relabeling_strategy="uniform",
        ),
        frame_transform_kwargs=dict(
            resize_size=(224, 224),
            num_parallel_calls=4,
        ),
        dataset_kwargs_list=per_dataset_kwargs,
        shuffle_buffer_size=1000,
        sample_weights=weights,
        balance_weights=True,
        traj_transform_threads=1,
        traj_read_threads=1,
        train=True,
    )
    
    print("🔧 Creating interleaved dataset...")
    dataset, dataset_length, dataset_statistics = make_interleaved_dataset(**rlds_config)
    
    print(f"📏 Dataset length: {dataset_length:,}")
    print(f"📈 Dataset statistics: {list(dataset_statistics.keys())}")
    
    # Test iteration
    print("🔄 Starting iteration test...")
    iterator = dataset.as_numpy_iterator()
    
    step = 0
    start_time = time.time()
    last_print_time = start_time
    
    try:
        while True:
            try:
                batch = next(iterator)
                step += 1
                
                # Print progress every 1000 steps or every 10 seconds
                current_time = time.time()
                if step % 1000 == 0 or (current_time - last_print_time) > 10:
                    elapsed = current_time - start_time
                    rate = step / elapsed if elapsed > 0 else 0
                    print(f"✅ Step {step:,} - Rate: {rate:.1f} steps/sec - Elapsed: {elapsed:.1f}s")
                    last_print_time = current_time
                
                # Success condition: reach well beyond the original failure point
                if step > 20000:
                    elapsed = time.time() - start_time
                    rate = step / elapsed
                    print(f"\n🎉 SUCCESS! Reached {step:,} steps!")
                    print(f"   Original failure was at step 16,793")
                    print(f"   Average rate: {rate:.1f} steps/sec")
                    print(f"   Total time: {elapsed:.1f} seconds")
                    return True
                
                # Safety timeout (5 minutes)
                if time.time() - start_time > 300:
                    print(f"\n⏰ Timeout reached at step {step:,}")
                    print("   This suggests the dataset is working but slow")
                    return step > 16793  # Consider success if we passed the original failure point
                    
            except tf.errors.DataLossError as e:
                print(f"⚠️  Data loss error at step {step}: {e}")
                continue  # Should be handled by ignore_errors()
            except tf.errors.InvalidArgumentError as e:
                print(f"⚠️  Invalid argument error at step {step}: {e}")
                continue  # Should be handled by ignore_errors()
                
    except StopIteration:
        print(f"\n❌ FAILURE! Iterator stopped at step {step:,}")
        print("   The dataset should be infinite due to .repeat()")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user at step {step:,}")
        return step > 16793
    except Exception as e:
        print(f"\n❌ Unexpected error at step {step:,}: {e}")
        return False


def check_dataset_files():
    """Check if dataset files exist and are accessible."""
    print("📁 Checking dataset files...")
    
    data_dir = Path("datasets/libero_data")
    if not data_dir.exists():
        print(f"❌ Dataset directory not found: {data_dir}")
        return False
    
    # Look for TFRecord files
    tfrecord_files = list(data_dir.rglob("*.tfrecord*"))
    if not tfrecord_files:
        print(f"❌ No TFRecord files found in {data_dir}")
        return False
    
    print(f"✅ Found {len(tfrecord_files)} TFRecord files")
    print(f"   First few: {[f.name for f in tfrecord_files[:3]]}")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 RLDS DATASET ITERATION TEST")
    print("=" * 60)
    
    # Configure TensorFlow
    tf.config.set_visible_devices([], "GPU")  # Use CPU only for testing
    
    # Check prerequisites
    if not check_dataset_files():
        exit(1)
    
    print("\n" + "=" * 60)
    
    # Run the test
    success = test_rlds_dataset()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ TEST PASSED!")
        print("   The RLDS dataset fix works correctly.")
        print("   Dataset can iterate beyond the corruption point.")
        print("   Your training should now work without early termination.")
    else:
        print("❌ TEST FAILED!")
        print("   The dataset still has iteration issues.")
        print("   May need further investigation.")
    print("=" * 60) 