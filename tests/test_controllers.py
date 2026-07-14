import unittest

from controllers import (
    bang_bang_control,
    clip_torque,
    p_control,
    pd_control,
)


class ControllerTests(unittest.TestCase):

    def test_bang_bang_positive_direction(self) -> None:
        torque = bang_bang_control(
            current_position=0.0,
            target_position=1.0,
            max_torque=3.0,
        )
        self.assertEqual(torque, 3.0)

    def test_bang_bang_negative_direction(self) -> None:
        torque = bang_bang_control(
            current_position=1.0,
            target_position=0.0,
            max_torque=3.0,
        )
        self.assertEqual(torque, -3.0)

    def test_bang_bang_deadband(self) -> None:
        torque = bang_bang_control(
            current_position=0.99,
            target_position=1.0,
            max_torque=3.0,
            deadband=0.02,
        )
        self.assertEqual(torque, 0.0)

    def test_p_control_sign(self) -> None:
        torque = p_control(
            current_position=0.5,
            target_position=1.0,
            kp=2.0,
        )
        self.assertAlmostEqual(torque, 1.0)

    def test_pd_derivative_term_opposes_motion(self) -> None:
        torque = pd_control(
            current_position=0.0,
            current_velocity=2.0,
            target_position=0.0,
            target_velocity=0.0,
            kp=2.0,
            kd=0.3,
        )
        self.assertAlmostEqual(torque, -0.6)

    def test_pd_combines_position_and_velocity_error(self) -> None:
        torque = pd_control(
            current_position=0.5,
            current_velocity=0.2,
            target_position=1.0,
            target_velocity=0.0,
            kp=2.0,
            kd=0.5,
        )

        # Position term: 2.0 × (1.0 - 0.5) = 1.0
        # Velocity term: 0.5 × (0.0 - 0.2) = -0.1
        self.assertAlmostEqual(torque, 0.9)

    def test_torque_clipping(self) -> None:
        self.assertEqual(clip_torque(5.0, 3.0), 3.0)
        self.assertEqual(clip_torque(-5.0, 3.0), -3.0)
        self.assertEqual(clip_torque(2.0, 3.0), 2.0)


if __name__ == "__main__":
    unittest.main()