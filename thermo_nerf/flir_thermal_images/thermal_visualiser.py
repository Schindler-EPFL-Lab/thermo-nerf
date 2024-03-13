import matplotlib.pyplot as plt


class ThermalVisualiser:
    """
    Thermal image visualiser to display the temperature upon hovering.

    This class deals with the greyscaled thermal images.
    Given a thermal images and its related temperature bounds values,
    it directly maps the pixel values to the temperature values and
    displays the temperature value upon hovering on a pixel.
    """

    def __init__(self, thermal_image, max_temperature, min_temperature) -> None:
        self.thermal_image = thermal_image
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.thermal_image, cmap="gray")

        self.annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(-20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        self.annot.set_visible(False)

    def update_temperature(self, pixel_value: float) -> float:
        """
        Update the temperature based on the `pixel_value`
        by denormalising the pixel value.

        :returns: temperature in degrees celsius

        """
        temperature = (pixel_value) * (
            self.max_temperature - self.min_temperature
        ) + self.min_temperature
        return temperature

    def hover(self, event: object) -> None:
        """
        Hover `event` handler to update the annotation box
        to display a temperature value.
        """

        if event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            pixel_value = self.thermal_image[y, x]
            temperature = self.update_temperature(pixel_value)

            self.annot.xy = (x, y)
            self.annot.set_text(f"Temperature: {temperature:.2f}")
            self.annot.get_bbox_patch().set_alpha(0.4)
            self.annot.set_visible(True)
            self.fig.canvas.draw_idle()
