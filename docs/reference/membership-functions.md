# Membership function API

Import these classes from `mostly.membership_functions`. Each concrete
membership function is a Pydantic model: pass its fields as keyword arguments,
then call the model with a finite scalar. Invalid field values, or call inputs
that cannot be parsed as finite scalars, raise `pydantic.ValidationError`.
Custom subclasses of `MembershipFunction` define their own validation.

The `diff` method of `MFSigmoidalCombined` can produce negative values. Some
implementations can overflow or return `NaN` for extreme finite inputs.

## Reference pages

- [MembershipFunction](./membership-functions/base.md)
- [MFCrisp](./membership-functions/crisp.md)
- [MFTriangular](./membership-functions/triangular.md)
- [MFTrapezoidal](./membership-functions/trapezoidal.md)
- [MFGaussian](./membership-functions/gaussian.md)
- [MFBimodalGaussian](./membership-functions/bimodal-gaussian.md)
- [MFGeneralizedBell](./membership-functions/generalized-bell.md)
- [MFPolynomialS](./membership-functions/polynomial-s.md)
- [MFPolynomialZ](./membership-functions/polynomial-z.md)
- [MFPolynomialCombined](./membership-functions/polynomial-combined.md)
- [MFSigmoidal](./membership-functions/sigmoidal.md)
- [MFSigmoidalCombined](./membership-functions/sigmoidal-combined.md)
