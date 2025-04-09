// Prevent duplicate rankings in the index page
document.addEventListener('DOMContentLoaded', function() {
    // Get all rank select elements
    const rankSelects = document.querySelectorAll('.rank-select');
    
    // Add change event listeners to each select
    rankSelects.forEach(select => {
        select.addEventListener('change', function() {
            // Get the selected value
            const selectedValue = this.value;
            const currentSelectId = this.id;
            
            // Check if any other select has the same value
            rankSelects.forEach(otherSelect => {
                if (otherSelect.id !== currentSelectId && otherSelect.value === selectedValue) {
                    // Find previous value of the current select
                    const previousValues = {};
                    rankSelects.forEach(s => {
                        previousValues[s.id] = s.value;
                    });
                    
                    // Swap values between the selects
                    const otherPreviousValue = previousValues[otherSelect.id];
                    otherSelect.value = otherPreviousValue;
                }
            });
        });
    });
});