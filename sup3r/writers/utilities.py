"""Basic objects that can cache rasterized / derived data."""

import logging
import os

logger = logging.getLogger(__name__)


def _check_for_cache(features, cache_kwargs):
    """Check if features are available in cache and return available
    files"""
    cache_kwargs = cache_kwargs or {}
    cache_pattern = cache_kwargs.get('cache_pattern', None)
    cached_files = []
    cached_features = []
    missing_files = []
    missing_features = features
    if cache_pattern is not None:
        cached_files = [
            cache_pattern.format(feature=f)
            for f in features
            if os.path.exists(cache_pattern.format(feature=f))
        ]
        cached_features = [
            f
            for f in features
            if os.path.exists(cache_pattern.format(feature=f))
        ]
        missing_features = list(set(features) - set(cached_features))
        missing_files = [
            cache_pattern.format(feature=f) for f in missing_features
        ]

    if cached_files:
        logger.debug(
            'Found cache files for %s with file pattern: %s',
            cached_features, cache_pattern
        )
    return cached_files, cached_features, missing_files, missing_features
