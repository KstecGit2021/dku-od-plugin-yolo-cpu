var app = angular.module('detectionRecipe.retrain', []);

app.controller('retrainRecipeController', function($scope) {
    $scope.optimizerOptions = [
        ["Adam", "adam"],
        ["SGD", "sgd"]
    ];

    var initVariable = function(varName, initValue) {
        if ($scope.config[varName] == undefined) {
            $scope.config[varName] = initValue;
        }
    };


    var retrieveInfoRetrain = function() {
        $scope.callPythonDo({method: "get-dataset-info"}).then(function(data) {
            $scope.canUseGPU = data["can_use_gpu"];
            $scope.labelColumns = data["columns"];
            
            initVariable("min_side", data["min_side"]);
            initVariable("max_side", data["max_side"]);
            
            $scope.finishedLoading = true;
        }, function(data) {
            $scope.finishedLoading = true;
        });
    };

    var initVariables = function() {
        initVariable("val_split", 0.8);
        initVariable("should_use_gpu", false);
        initVariable("gpu_allocation", 0.5);
        initVariable("list_gpu", "0");
        initVariable("lr", 0.00001);
        initVariable("nb_epochs", 10);
        initVariable("tensorboard", false);
        initVariable("optimizer", "adam");
        initVariable("freeze", true);
        initVariable("epochs", 10);
        initVariable("flip_x", 0.5);
        initVariable("flip_y", 0.0);
        initVariable("min_rotation", -0.1);
        initVariable("max_rotation", 0.1);
        initVariable("min_trans", -0.1);
        initVariable("max_trans", 0.1);
        initVariable("min_shear", -0.1);
        initVariable("max_shear", 0.1);
        initVariable("min_scaling", 0.9);
        initVariable("max_scaling", 1.1);
        initVariable("reducelr", false);
        initVariable("reducelr_patience", 2);
        initVariable("reducelr_factor", 0.1);
        initVariable("single_column_data", false);
    };

    var init = function() {
        initVariables();
        retrieveInfoRetrain();
    };

    init();
});
