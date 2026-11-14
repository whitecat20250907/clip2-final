from __future__ import annotations
import numpy as np
import soundfile as sf
import soxr
from librosa import util
from librosa.util.exceptions import ParameterError
def load(
    path: str | int | os.PathLike[Any] | sf.SoundFile | BinaryIO,
    *,
    sr: float | None = 22050,
    mono: bool = True,
    offset: float = 0.0,
    duration: float | None = None,
    dtype: DTypeLike = np.float32,
    res_type: str = "soxr_hq",
) -> tuple[np.ndarray, int | float]:
    """Load an audio file as a floating point time series.

    Audio will be automatically resampled to the given rate
    (default ``sr=22050``).

    To preserve the native sampling rate of the file, use ``sr=None``.

    Parameters
    ----------
    path : str, int, pathlib.Path, soundfile.SoundFile, or file-like object
        path to the input file.

        Any codec supported by `soundfile` will work.

        Any string file paths, or any object implementing Python's
        file interface (e.g. `pathlib.Path`) are supported as `path`.

        If the codec is supported by `soundfile`, then `path` can also be
        an open file descriptor (int) or an existing `soundfile.SoundFile` object.

    sr : number > 0 [scalar]
        target sampling rate

        'None' uses the native sampling rate

    mono : bool
        convert signal to mono

    offset : float
        start reading after this time (in seconds).

        If negative, it will be interpreted relative to the end of the file.

    duration : float
        only load up to this much audio (in seconds)

    dtype : numeric type
        data type of ``y``

    res_type : str
        resample type (see note)

        .. note::
            By default, this uses `soxr`'s high-quality mode ('HQ').

            For alternative resampling modes, see `resample`

    Returns
    -------
    y : np.ndarray [shape=(n,) or (..., n)]
        audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of ``y``

    Examples
    --------
    >>> # Load an ogg vorbis file
    >>> filename = librosa.ex('trumpet')
    >>> y, sr = librosa.load(filename)
    >>> y
    array([-1.407e-03, -4.461e-04, ..., -3.042e-05,  1.277e-05],
          dtype=float32)
    >>> sr
    22050

    >>> # Load a file and resample to 11 KHz
    >>> filename = librosa.ex('trumpet')
    >>> y, sr = librosa.load(filename, sr=11025)
    >>> y
    array([-8.746e-04, -3.363e-04, ..., -1.301e-05,  0.000e+00],
          dtype=float32)
    >>> sr
    11025

    >>> # Load 5 seconds of a file, starting 15 seconds in
    >>> filename = librosa.ex('brahms')
    >>> y, sr = librosa.load(filename, offset=15.0, duration=5.0)
    >>> y
    array([0.146, 0.144, ..., 0.128, 0.015], dtype=float32)
    >>> sr
    22050

    >>> # Load using an already open SoundFile object
    >>> import soundfile
    >>> sfo = soundfile.SoundFile(librosa.ex('brahms'))
    >>> y, sr = librosa.load(sfo)
    """
    y, sr_native = __soundfile_load(path, offset, duration, dtype)

    # Final cleanup for dtype and contiguity
    if mono:
        y = to_mono(y)

    if sr is not None:
        y = resample(y, orig_sr=sr_native, target_sr=sr, res_type=res_type)

    else:
        sr = sr_native

    return y, sr

def __soundfile_load(path, offset, duration, dtype):
    """Load an audio buffer using soundfile."""
    if isinstance(path, sf.SoundFile):
        # If the user passed an existing soundfile object,
        # we can use it directly
        context = path
    else:
        # Otherwise, create the soundfile object
        context = sf.SoundFile(path)

    with context as sf_desc:
        sr_native = sf_desc.samplerate
        if offset != 0:
            if offset > 0:
                # Seek to the start of the target read
                sf_desc.seek(int(offset * sr_native))
            else:
                sf_desc.seek(-int(abs(offset) * sr_native), whence=sf.SEEK_END)

        if duration is not None:
            frame_duration = int(duration * sr_native)
        else:
            frame_duration = -1

        # Load the target number of frames, and transpose to match librosa form
        y = sf_desc.read(frames=frame_duration, dtype=dtype, always_2d=False).T

    return y, sr_native

def _init_output(
    out: np.ndarray | None, shape: tuple[int, ...], dtype: np.dtype
) -> np.ndarray:
    """Validate a caller-provided output buffer, or allocate a new one.

    If ``out`` is ``None``, return a new zero-filled array of the given shape
    and dtype.  Otherwise ``out`` must match ``shape`` exactly, and its dtype
    must be able to hold ``dtype`` without loss of precision (i.e. at least as
    precise as ``dtype``); it is zeroed in place and returned.  A caller that
    explicitly allocates a higher-precision buffer keeps that precision.
    """
    if out is None:
        return np.zeros(shape, dtype=dtype)
    if out.shape != shape:
        raise ParameterError(f"out has shape={out.shape}, but expected {shape}")
    if np.promote_types(out.dtype, dtype) != out.dtype:
        raise ParameterError(
            f"out has dtype={out.dtype}, which cannot hold the result dtype {dtype}"
        )
    out[...] = 0
    return out

def to_mono(
    *signals: np.ndarray,
    pad: bool = True,
    norm: bool = True,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Convert an audio signal to mono by averaging samples across channels.

    Parameters
    ----------
    *signals : np.ndarray [shape=(..., n)]
        One or more signals to convert to mono.
        Each input signal can be mono or multi-channel, and they need not all have
        identical lengths.

    pad : bool
        If `True`, pad the output to match the length of the longest input:
        `n_out = max(y.shape[-1] for y in signals)`.

        If `False`, trim the output to match the length of the shortest input:
        `n_out = min(y.shape[-1] for y in signals)`.

    norm : bool
        If `True` (default), signals are combined by averaging.

        If `False`, signals are combined by summing.

    out : np.ndarray [shape=(n_out,)] or None
        A pre-allocated output buffer.  If provided, it must match the shape
        ``(n_out,)``, and its dtype must be at least as precise as the result
        dtype; it is zeroed and filled in place and returned, avoiding a new
        allocation.  If ``None`` (default), a new array is allocated.

    Returns
    -------
    y_mono : np.ndarray [shape=(n_out,)]
        All signals combined together into a single mono signal.
        If ``out`` was provided, this is the same object as ``out``.

    Notes
    -----
    The dtype of the output signal will be the most general type that can
    accommodate all input signals.  If the input signals have different
    dtypes, the output will be promoted to the most general type.

    .. warning:: Any input signal with more than one channel will be
      downmixed to mono prior to being combined with other signals.
      This means that the following are generally not equivalent
      for multi-channel inputs `y1` and `y2` when `norm=True`:

        >>> y_mono = librosa.to_mono(y1, y2)
        >>> y_mono = librosa.to_mono(np.vstack((y1, y2)))

    See Also
    --------
    to_stereo
    to_multi
    util.fix_length

    Examples
    --------
    Downmix a stereo input

    >>> y, sr = librosa.loadx('trumpet', mono=False)
    >>> y.shape
    (2, 117601)
    >>> y_mono = librosa.to_mono(y)
    >>> y_mono.shape
    (117601,)

    Mix three independent mono inputs with different lengths

    >>> y_440 = librosa.tone(440.0, sr=22050, duration=1.0)
    >>> y_550 = librosa.tone(550.0, sr=22050, duration=1.5)
    >>> y_660 = librosa.tone(660.0, sr=22050, duration=2.0)
    >>> y_mono = librosa.to_mono(y_440, y_550, y_660, pad=True)
    >>> y_mono.shape
    (44100,)
    """
    # Validate the buffer(s) and identify lengths, dtypes, etc
    if not signals:
        raise ParameterError("At least one signal must be provided to `to_mono`.")
    n_min, n_max = signals[0].shape[-1], signals[0].shape[-1]
    dtype = signals[0].dtype
    for y in signals:
        util.valid_audio(y)
        n_min = min(n_min, y.shape[-1])
        n_max = max(n_max, y.shape[-1])
        dtype = np.promote_types(dtype, y.dtype)

    # Allocate and initialize the output buffer
    if pad:
        size = n_max
    else:
        size = n_min

    output = _init_output(out, (size,), dtype)

    if norm:
        combine = np.mean
    else:
        combine = np.sum

    # Now, average the signals together
    for y in signals:
        output += util.fix_length(
            combine(y, axis=tuple(range(y.ndim - 1))), size=output.shape[-1], axis=-1
        )
    if norm:
        # Divide by the number of input signals given
        output /= len(signals)

    return output

def resample(
    y: np.ndarray,
    *,
    orig_sr: float,
    target_sr: float,
    res_type: str = "soxr_hq",
    fix: bool = True,
    scale: bool = False,
    axis: int = -1,
    **kwargs: Any,
) -> np.ndarray:
    """Resample a time series from orig_sr to target_sr

    By default, this uses a high-quality method (`soxr_hq`) for band-limited sinc
    interpolation. The alternate ``res_type`` values listed below offer different
    trade-offs of speed and quality.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n, ...)]
        audio time series, with `n` samples along the specified axis.

    orig_sr : number > 0 [scalar]
        original sampling rate of ``y``

    target_sr : number > 0 [scalar]
        target sampling rate

    res_type : str (default: `soxr_hq`)
        resample type

        'soxr_vhq', 'soxr_hq', 'soxr_mq' or 'soxr_lq'
            `soxr` Very high-, High-, Medium-, Low-quality FFT-based bandlimited interpolation.
            ``'soxr_hq'`` is the default setting of `soxr`.
        'soxr_qq'
            `soxr` Quick cubic interpolation (very fast, but not bandlimited)
        'kaiser_best'
            `resampy` high-quality mode
        'kaiser_fast'
            `resampy` faster method
        'fft' or 'scipy'
            `scipy.signal.resample` Fourier method.
        'polyphase'
            `scipy.signal.resample_poly` polyphase filtering. (fast)
        'linear'
            `samplerate` linear interpolation. (very fast, but not bandlimited)
        'zero_order_hold'
            `samplerate` repeat the last value between samples. (very fast, but not bandlimited)
        'sinc_best', 'sinc_medium' or 'sinc_fastest'
            `samplerate` high-, medium-, and low-quality bandlimited sinc interpolation.

        .. note::
            Not all options yield a bandlimited interpolator. If you use `soxr_qq`, `polyphase`,
            `linear`, or `zero_order_hold`, you need to be aware of possible aliasing effects.

        .. note::
            `samplerate` and `resampy` are not installed with `librosa`.
            To use `samplerate` or `resampy`, they should be installed manually::

                $ pip install samplerate
                $ pip install resampy

        .. note::
            When using ``res_type='polyphase'``, only integer sampling rates are
            supported.

    fix : bool
        adjust the length of the resampled signal to be of size exactly
        ``ceil(target_sr * len(y) / orig_sr)``

    scale : bool
        Scale the resampled signal so that ``y`` and ``y_hat`` have approximately
        equal total energy.

    axis : int
        The target axis along which to resample.  Defaults to the trailing axis.

    **kwargs : additional keyword arguments
        If ``fix==True``, additional keyword arguments to pass to
        `librosa.util.fix_length`.

    Returns
    -------
    y_hat : np.ndarray [shape=(..., n * target_sr / orig_sr, ...)]
        ``y`` resampled from ``orig_sr`` to ``target_sr`` along the target axis

    Raises
    ------
    ParameterError
        If ``res_type='polyphase'`` and ``orig_sr`` or ``target_sr`` are not both
        integer-valued.

    See Also
    --------
    librosa.util.fix_length
    scipy.signal.resample
    resampy
    samplerate.converters.resample
    soxr.resample

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    Downsample from 22 KHz to 8 KHz

    >>> y, sr = librosa.loadx('trumpet', sr=22050)
    >>> y_8k = librosa.resample(y, orig_sr=sr, target_sr=8000)
    >>> y.shape, y_8k.shape
    ((117601,), (42668,))
    """
    # First, validate the audio buffer
    util.valid_audio(y)

    if orig_sr == target_sr:
        return y

    ratio = float(target_sr) / orig_sr

    n_samples = int(np.ceil(y.shape[axis] * ratio))

    if res_type in ("scipy", "fft"):
        import scipy.signal

        y_hat = scipy.signal.resample(y, n_samples, axis=axis)
    elif res_type == "polyphase":
        if int(orig_sr) != orig_sr or int(target_sr) != target_sr:
            raise ParameterError(
                "polyphase resampling is only supported for integer-valued sampling rates."
            )

        import scipy.signal

        # For polyphase resampling, we need up- and down-sampling ratios
        # We can get those from the greatest common divisor of the rates
        # as long as the rates are integrable
        orig_sr = int(orig_sr)
        target_sr = int(target_sr)
        gcd = np.gcd(orig_sr, target_sr)
        y_hat = scipy.signal.resample_poly(
            y, target_sr // gcd, orig_sr // gcd, axis=axis
        )
    elif res_type in (
        "linear",
        "zero_order_hold",
        "sinc_best",
        "sinc_fastest",
        "sinc_medium",
    ):
        # Use numpy to vectorize the resampler along the target axis
        # This is because samplerate does not support ndim>2 generally.
        y_hat = np.apply_along_axis(
            samplerate.resample, axis=axis, arr=y, ratio=ratio, converter_type=res_type
        )
    elif res_type.startswith("soxr"):
        # Use numpy to vectorize the resampler along the target axis
        # This is because soxr does not support ndim>2 generally.
        y_hat = np.apply_along_axis(
            soxr.resample,
            axis=axis,
            arr=y,
            in_rate=orig_sr,
            out_rate=target_sr,
            quality=res_type,
        )
    else:
        y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=axis)

    if fix:
        y_hat = util.fix_length(y_hat, size=n_samples, axis=axis, **kwargs)

    if scale:
        y_hat /= np.sqrt(ratio)

    # Match dtypes
    return np.asarray(y_hat, dtype=y.dtype)