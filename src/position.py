import numpy as np


class Vector(np.ndarray):
    """
    A NumPy-based 3D vector class that supports array operations and Euclidean space functionality.

    Features:
    - Component access via .x, .y, .z properties
    - Full NumPy array compatibility
    - Vector-specific operations (dot product, cross product, normalization)
    - Type preservation through arithmetic operations
    - Support for both single vectors and arrays of vectors

    Shape Requirements:
    - Last dimension must be size 3 (x, y, z components)
    - Valid shapes: (3,), (N, 3), (N, M, 3), etc.

    Example Usage:
    >>> v = Vector([1, 2, 3])
    >>> u = Vector([[4, 5, 6], [7, 8, 9]])
    >>> v + u
    Vector([[ 5.,  7.,  9.],
            [ 8., 10., 12.]])
    >>> v @ u  # Dot product
    array([32., 50.])
    """

    # ---------- Magic methods ----------
    def __new__(cls, input_array):
        """Create a new Vector instance from an array-like input."""
        obj = np.asarray(input_array, dtype=np.float64).view(cls)

        if obj.shape[-1] != 3:
            raise ValueError(
                f"Last dimension must be size 3 (x, y, z). Got shape {obj.shape}"
            )
        return obj

    def __array_finalize__(self, obj):
        """Ensure proper initialization during array operations."""
        if obj is None:
            return

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle NumPy universal functions to preserve Vector type."""
        # Convert Vector instances to base ndarray for computation
        inputs = tuple(
            x.view(np.ndarray) if isinstance(x, Vector) else x for x in inputs
        )

        # Convert Vector instances in 'out' to ndarray views
        if "out" in kwargs:
            kwargs["out"] = tuple(
                o.view(np.ndarray) if isinstance(o, Vector) else o
                for o in kwargs["out"]
            )

        # Perform the operation using the superclass implementation
        results = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

        # Handle cases where operation isn't supported
        if results is NotImplemented:
            return NotImplemented

        # Convert back to Vector if result has 3 components in last dimension
        if isinstance(results, np.ndarray) and results.shape[-1] == 3:
            return results.view(Vector)
        return results

    def __repr__(self):
        """Show data without double 'Vector' wrapping."""
        data_str = np.array2string(
            self.view(np.ndarray), separator=", ", prefix="Vector(", suffix=")"
        )
        return f"Vector({data_str})"

    def __str__(self):
        """User-friendly string representation."""
        return f"Vector:\n{super().__str__()}"

    def __matmul__(self, other):
        """Enable @ operator for dot product."""
        return self.dot(other)

    # ---------- Properties ----------
    @property
    def x(self):
        """X-component(s) of the vector(s) as a standard ndarray."""
        return self[..., 0].view(np.ndarray)

    @x.setter
    def x(self, value):
        self[..., 0] = value

    @property
    def y(self):
        """Y-component(s) of the vector(s) as a standard ndarray."""
        return self[..., 1].view(np.ndarray)

    @y.setter
    def y(self, value):
        self[..., 1] = value

    @property
    def z(self):
        """Z-component(s) of the vector(s) as a standard ndarray."""
        return self[..., 2].view(np.ndarray)

    @z.setter
    def z(self, value):
        self[..., 2] = value

    # ---------- Static methods ----------
    @staticmethod
    def zero(shape=()):
        """Create zero vector(s) of specified shape."""
        return Vector(np.zeros(shape + (3,)))

    @staticmethod
    def random_gauss(shape=(), scale=1):
        """Create random vector(s) with components in [-scale, scale]."""
        return Vector(np.random.normal(loc=0, scale=scale, size=shape + (3,)))

    # ---------- Other methods ----------
    def norm(self, axis=-1, keepdims=False):
        """Compute the magnitude (L2 norm) of the vector(s)."""
        return np.linalg.norm(self, axis=axis, keepdims=keepdims)

    def normalized(self):
        """Return unit vector(s) in the same direction."""
        return self / self.norm(keepdims=True)

    def dot(self, other):
        """Compute dot product with another vector/array of vectors."""
        return np.sum(self * other, axis=-1)

    def cross(self, other):
        """Compute cross product with another vector/array of vectors."""
        return np.cross(self, other).view(Vector)

    def angle_between(self, other, degrees=False):
        """Compute angle between vectors."""
        cos_theta = self.dot(other) / (self.norm() * other.norm())
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(angle) if degrees else angle
