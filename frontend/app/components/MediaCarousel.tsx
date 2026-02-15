"use client";

import Image from "next/image";
import { Carousel } from "antd";

export type CarouselItem = {
  src: string;
  alt: string;
  caption: string;
};

export default function MediaCarousel({ items }: { items: CarouselItem[] }) {
  return (
    <section className="media-carousel-wrap">
      <Carousel autoplay autoplaySpeed={3800} effect="fade" dots={{ className: "media-carousel-dots" }}>
        {items.map((item) => (
          <div key={`${item.src}-${item.caption}`}>
            <div className="media-carousel-slide">
              <Image src={item.src} alt={item.alt} fill sizes="(max-width:900px) 100vw, 1200px" priority />
              <div className="media-carousel-mask" />
              <div className="media-carousel-caption">{item.caption}</div>
            </div>
          </div>
        ))}
      </Carousel>
    </section>
  );
}
