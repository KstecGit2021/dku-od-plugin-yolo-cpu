var app = angular.module('detectionRecipe.draw', []);

app.controller('drawRecipeController', function($scope) {

    var initVariable = function(varName, initValue) {
        if ($scope.config[varName] == undefined) {
            $scope.config[varName] = initValue;
        }
    };

    var retrieveInfoRetrain = function() {
        $scope.callPythonDo({method: "get-confidence"}).then(function(data) {
            $scope.hasConfidence = data["has_confidence"];
            $scope.finishedLoading = true;
        }, function(data) {
            $scope.finishedLoading = true;
        });
    };

    var initVariables = function() {
        initVariable("draw_label", true);
        initVariable("draw_confidence", false);
    };

    var init = function() {
        initVariables();
        retrieveInfoRetrain();
    };

    init();
});