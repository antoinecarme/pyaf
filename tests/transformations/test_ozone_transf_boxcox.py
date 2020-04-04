
import tests.transformations.test_ozone_transf_generic as gen
import warnings;

with warnings.catch_warnings():
    # warnings.simplefilter("error");
    gen.test_transformation('BoxCox');
