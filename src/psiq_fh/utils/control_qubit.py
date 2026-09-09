"""Module for control qubits (i.e. phase qubits)."""

from psiqdk.workbench import Qubits


class ControlQubit(Qubits):
    """Type for a control qubit.

    Note:
        - This inherits all the standard Qubits methods but allows us to explicitly
          mark qubits that are used as controls.
        - We override rotation operations so that we can (optionally) either defer them
          to the end of a series of such rotations or skip them entirely.
    """

    def __init__(
        self,
        *args,
        defer_rotations=False,
        skip_rotations=False,
        **kwargs,
    ):
        """Initialize a control qubit.

        Args:
            defer_rotations (bool): If True, tracks rotations instead of applying them
            skip_rotations (bool): If True, skips rotations instead of applying them
            args (list): args
            kwargs (dict): keyword args
        """
        if defer_rotations and skip_rotations:
            raise ValueError("defer_phase_rotations and skip_rotations are mutually exclusive.")

        self.defer_rotations = defer_rotations
        self.skip_rotations = skip_rotations
        self.phase_rots = []
        self.rz_rots = []

        super().__init__(*args, **kwargs)

        if self.num_qubits != 1:
            raise RuntimeError("ControlQubit is only supported for a single qubit control qubit!")

    def phase(self, theta, *args, **kwargs):
        """Apply phase rotation.

        Args:
            theta (float or Units): Rotation angle, in degrees by default
            args (list): args
            kwargs (dict): keyword args
        """
        # Completely ignore rotations
        if self.skip_rotations:
            return

        # Accumulate for later resolution
        if self.defer_rotations:
            self.phase_rots.append(float(theta))
            return

        # Apply immediately
        super().phase(theta, *args, **kwargs)

    def reflect(self, theta: float = None, *args, **kwargs):
        """Apply reflect operation (which is a phase gate for single qubit).

        Args:
            basis (Literal): string indicating basis for reflect operation
            theta (float or Units): rotation angle, in degrees by default
            args (list): args
            kwargs (dict): keyword args

        Notes:
            - This means that ControlQubit has to be a single qubit currently, since
              it's included in the phase rotation list.
        """
        # Completely ignore rotations
        if self.skip_rotations:
            return

        # Accumulate for later resolution
        if self.defer_rotations:
            self.phase_rots.append(float(theta))
            return

        # Apply immediately
        super().reflect("z", theta, *args, **kwargs)

    def rz(self, theta, *args, **kwargs):
        """Apply RZ rotation.

        Args:
            theta (float or Units): Rotation angle, in degrees by default
            args (list): args
            kwargs (dict): keyword args
        """
        if self.skip_rotations:
            return

        if self.defer_rotations:
            self.rz_rots.append(float(theta))
            return

        super().rz(theta, *args, **kwargs)

    def resolve_rotations(self, merge_phase_and_rz=False, *args, **kwargs):
        """Resolve rotations by either executing them as one (or two) merged rotations
           or skipping them.

        Args:
            merge_phase_and_rz (bool): If True, merges phase and RZ rotations into one
                                       RZ rotation (which can be done up to a global phase)
            args (list): args
            kwargs (dict): keyword args
        """
        # Nothing to do if skipping
        if self.skip_rotations:
            return

        # Note: overall unitary could be off by a global phase
        if not merge_phase_and_rz:
            # Apply accumulated rotation
            if self.defer_rotations and self.phase_rots:
                super().phase(sum(self.phase_rots), *args, **kwargs)
                self.phase_rots.clear()
        else:
            self.rz_rots += self.phase_rots
            self.phase_rots.clear()

        # Apply accumulated rotation
        if self.defer_rotations and self.rz_rots:
            super().rz(sum(self.rz_rots), *args, **kwargs)
            self.rz_rots.clear()


class DirectionalControlQubit(ControlQubit):
    """Wrapper for directional control qubit."""

    @classmethod
    def from_control(cls, q: ControlQubit):
        """Helper to retain lists of rotations when they are deferred."""
        new_ctrl = DirectionalControlQubit(q, defer_rotations=q.defer_rotations, skip_rotations=q.skip_rotations)
        new_ctrl.phase_rots = q.phase_rots
        new_ctrl.rz_rots = q.rz_rots
        return new_ctrl
